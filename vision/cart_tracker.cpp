/*
 * Autonomous Shopping Cart — Vision Tracker
 *
 * Detects people (and optionally carts) using Darknet yolov4-tiny,
 * tracks them with the built-in Kalman filter, and outputs distance
 * and angle to the target (closest + most-centred person) every frame.
 *
 * Usage:
 *   ./cart_tracker <image|video|webcam_id>
 *
 * Models expected next to the binary (or in vision/):
 *   yolov4-tiny.cfg + yolov4-tiny.weights + coco.names  — person detection
 *   cart_darknet.cfg + cart_darknet.weights + cart.names — cart detection (optional)
 */

#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "darknet.hpp"
#include "yolo_v2_class.hpp"   // track_kalman_t, bbox_t

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Camera / geometry calibration
// ---------------------------------------------------------------------------

static constexpr float H_FOV_DEG         = 54.0f;
static constexpr float PERSON_HEIGHT_M   = 1.7f;
static constexpr float DISTANCE_OFFSET_M = 3.0f;
static constexpr float DISTANCE_SCALE    = 0.5f;
static constexpr float ANGLE_SCALE       = 2.0f;

// Target selection weights
static constexpr float TARGET_DIST_WEIGHT  = 1.0f;
static constexpr float TARGET_ANGLE_WEIGHT = 0.3f;

// EMA smoothing factor for distance (lower = smoother)
static constexpr float DIST_EMA_ALPHA = 0.4f;

// Detection thresholds
static constexpr float PERSON_CONF = 0.45f;
static constexpr float CART_CONF   = 0.30f;

// Overhead map
static constexpr int   MAP_SIZE    = 500;
static constexpr float MAP_RANGE_M = 10.0f;

// Class IDs (internal)
static constexpr unsigned int CLS_PERSON = 0;
static constexpr unsigned int CLS_CART   = 1;

// Colors (BGR)
static const cv::Scalar COLOR_TARGET  (  0, 255,   0);
static const cv::Scalar COLOR_OBSTACLE(  0,   0, 255);
static const cv::Scalar COLOR_GRID    ( 50,  50,  50);
static const cv::Scalar COLOR_FOV     ( 40,  40,  40);

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

static float focal_length_px(float dim, float fov_deg)
{
    return (dim / 2.0f) / std::tan(fov_deg * M_PI / 360.0f);
}

static float estimate_distance(float bbox_h, float bbox_cx, float img_h, float img_w)
{
    const float v_fov     = H_FOV_DEG * (img_h / img_w);
    const float fl_v      = focal_length_px(img_h, v_fov);
    const float raw_depth = (PERSON_HEIGHT_M * fl_v) / bbox_h;
    const float raw_angle = std::atan((bbox_cx - img_w / 2.0f) / focal_length_px(img_w, H_FOV_DEG));
    const float slant     = raw_depth / std::cos(raw_angle);
    return std::max(0.0f, (slant - DISTANCE_OFFSET_M) * DISTANCE_SCALE);
}

static float estimate_angle(float bbox_cx, float img_w)
{
    const float fl = focal_length_px(img_w, H_FOV_DEG);
    return std::atan((bbox_cx - img_w / 2.0f) / fl) * (180.0f / M_PI) * ANGLE_SCALE;
}

// ---------------------------------------------------------------------------
// Detection struct (merged from all models)
// ---------------------------------------------------------------------------

struct Detection {
    cv::Rect      rect;
    float         conf;
    unsigned int  cls_id;   // CLS_PERSON or CLS_CART
    unsigned int  track_id; // assigned by Kalman tracker
};

// ---------------------------------------------------------------------------
// Target selection
// ---------------------------------------------------------------------------

static int find_target_idx(const std::vector<Detection> & dets, float img_w, float img_h)
{
    int   best_idx   = -1;
    float best_score = std::numeric_limits<float>::infinity();

    for (int i = 0; i < (int)dets.size(); ++i)
    {
        if (dets[i].cls_id != CLS_PERSON) continue;

        const float bbox_h  = dets[i].rect.height;
        const float bbox_cx = dets[i].rect.x + bbox_h / 2.0f;

        if (bbox_h <= 0) continue;

        const float dist  = estimate_distance(bbox_h, bbox_cx, img_h, img_w);
        const float angle = std::abs(estimate_angle(bbox_cx, img_w));
        const float score = TARGET_DIST_WEIGHT * dist + TARGET_ANGLE_WEIGHT * angle;

        if (score < best_score)
        {
            best_score = score;
            best_idx   = i;
        }
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// Overhead map
// ---------------------------------------------------------------------------

static cv::Mat draw_map(const std::vector<Detection> & dets, int target_idx,
                        float img_w, float img_h)
{
    cv::Mat img = cv::Mat::zeros(MAP_SIZE, MAP_SIZE, CV_8UC3);

    const int   cam_px = MAP_SIZE / 2;
    const int   cam_py = MAP_SIZE - 40;
    const float scale  = (MAP_SIZE - 60) / MAP_RANGE_M;

    auto to_px = [&](float dist, float angle_deg) -> cv::Point
    {
        const float rad = angle_deg * M_PI / 180.0f;
        return cv::Point(
            static_cast<int>(cam_px + dist * std::sin(rad) * scale),
            static_cast<int>(cam_py - dist * std::cos(rad) * scale)
        );
    };

    // Grid rings
    for (int r = 2; r <= (int)MAP_RANGE_M; r += 2)
    {
        cv::circle(img, {cam_px, cam_py}, static_cast<int>(r * scale), COLOR_GRID, 1);
        cv::putText(img, std::to_string(r) + "m",
                    {cam_px + (int)(r * scale) + 3, cam_py},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, {80,80,80}, 1);
    }

    // FOV lines
    for (int side : {-1, 1})
    {
        cv::line(img, {cam_px, cam_py}, to_px(MAP_RANGE_M, side * H_FOV_DEG / 2.0f), COLOR_FOV, 1);
    }
    cv::line(img, {cam_px, cam_py}, {cam_px, cam_py - (int)(MAP_RANGE_M * scale)}, COLOR_FOV, 1);

    // Detections
    for (int i = 0; i < (int)dets.size(); ++i)
    {
        const bool   is_target = (i == target_idx);
        const float  bbox_h    = dets[i].rect.height;
        const float  bbox_cx   = dets[i].rect.x + dets[i].rect.width / 2.0f;

        if (bbox_h <= 0) continue;

        const float     dist  = estimate_distance(bbox_h, bbox_cx, img_h, img_w);
        const float     angle = estimate_angle(bbox_cx, img_w);
        const cv::Point pt    = to_px(dist, angle);
        const cv::Scalar col  = is_target ? COLOR_TARGET : COLOR_OBSTACLE;

        std::string label = (is_target ? "TARGET" : (dets[i].cls_id == CLS_CART ? "CART" : "PERSON"));
        label += " ID" + std::to_string(dets[i].track_id);

        if (dets[i].cls_id == CLS_CART && !is_target)
        {
            cv::rectangle(img, {pt.x - 7, pt.y - 7}, {pt.x + 7, pt.y + 7}, col, -1);
            cv::putText(img, label, {pt.x + 9, pt.y + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.38, col, 1);
        }
        else
        {
            const int r = is_target ? 10 : 7;
            cv::circle(img, pt, r, col, -1);
            cv::putText(img, label, {pt.x + r + 2, pt.y + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.38, col, 1);
        }
    }

    // Camera icon
    const std::vector<cv::Point> cam_pts = {
        {cam_px,      cam_py - 12},
        {cam_px - 10, cam_py +  6},
        {cam_px + 10, cam_py +  6}
    };
    cv::fillPoly(img, std::vector<std::vector<cv::Point>>{cam_pts}, {200,200,200});
    cv::putText(img, "OVERHEAD MAP", {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {160,160,160}, 1);

    return img;
}

// ---------------------------------------------------------------------------
// Frame annotation
// ---------------------------------------------------------------------------

struct FrameResult {
    std::string role, label_id, class_tag;
    float       conf, dist, angle;
};

static std::vector<FrameResult> annotate_frame(
    cv::Mat                  & frame,
    const std::vector<Detection> & dets,
    std::map<unsigned int, float> & smooth_state,
    int                        target_idx)
{
    const float img_w = static_cast<float>(frame.cols);
    const float img_h = static_cast<float>(frame.rows);

    std::vector<FrameResult> rows;

    for (int i = 0; i < (int)dets.size(); ++i)
    {
        const bool  is_target = (i == target_idx);
        const auto & d        = dets[i];
        const auto   col      = is_target ? COLOR_TARGET : COLOR_OBSTACLE;

        const std::string role      = is_target ? "TARGET" : "OBSTACLE";
        const std::string label_id  = "ID" + std::to_string(d.track_id);
        const std::string class_tag = (d.cls_id == CLS_PERSON) ? "person" : "cart";

        const float bbox_h  = d.rect.height;
        const float bbox_cx = d.rect.x + d.rect.width / 2.0f;
        const float angle   = estimate_angle(bbox_cx, img_w);

        // EMA-smoothed distance
        float raw_dist = (bbox_h > 0) ? estimate_distance(bbox_h, bbox_cx, img_h, img_w) : 0.0f;
        float dist;
        if (d.track_id > 0)
        {
            auto it = smooth_state.find(d.track_id);
            float prev = (it != smooth_state.end()) ? it->second : raw_dist;
            dist = DIST_EMA_ALPHA * raw_dist + (1.0f - DIST_EMA_ALPHA) * prev;
            smooth_state[d.track_id] = dist;
        }
        else
        {
            dist = raw_dist;
        }

        // Bounding box
        cv::rectangle(frame, d.rect, col, is_target ? 3 : 2);

        // Label
        const std::string display_role = is_target ? role : class_tag;
        std::ostringstream lbl;
        lbl << display_role << " " << label_id << " "
            << std::fixed << std::setprecision(1) << dist << "m "
            << std::showpos << std::setprecision(1) << angle << "deg";
        const std::string label = lbl.str();

        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        const int top = std::max(d.rect.y - 10, ts.height + 4);
        cv::rectangle(frame, {d.rect.x, top - ts.height - 4}, {d.rect.x + ts.width, top}, col, -1);
        cv::putText(frame, label, {d.rect.x, top - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 2);

        rows.push_back({role, label_id, class_tag, d.conf, dist, angle});
    }

    return rows;
}

// ---------------------------------------------------------------------------
// Convert V3 Darknet::Predictions → bbox_t for the Kalman tracker
// ---------------------------------------------------------------------------

static std::vector<bbox_t> predictions_to_bbox(
    const Darknet::Predictions & preds,
    unsigned int                 cls_id,
    float                        conf_thresh,
    int                          person_class_idx)    // COCO index for "person" (-1 = accept all)
{
    std::vector<bbox_t> out;
    for (const auto & p : preds)
    {
        if (p.best_class < 0) continue;
        if (person_class_idx >= 0 && p.best_class != person_class_idx) continue;

        const float conf = p.prob.count(p.best_class) ? p.prob.at(p.best_class) : 0.0f;
        if (conf < conf_thresh) continue;

        bbox_t b{};
        b.x      = static_cast<unsigned int>(std::max(0, p.rect.x));
        b.y      = static_cast<unsigned int>(std::max(0, p.rect.y));
        b.w      = static_cast<unsigned int>(p.rect.width);
        b.h      = static_cast<unsigned int>(p.rect.height);
        b.prob   = conf;
        b.obj_id = cls_id;   // our internal class id (not the COCO one)
        out.push_back(b);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Model wrapper
// ---------------------------------------------------------------------------

struct Model {
    Darknet::NetworkPtr net   = nullptr;
    int                 person_class_idx = -1;  // set to 0 for COCO person model, -1 for cart model
    float               conf_thresh      = 0.45f;
    unsigned int        cls_id           = CLS_PERSON;
};

static Model load_model(const fs::path & cfg, const fs::path & names, const fs::path & weights,
                        float conf_thresh, unsigned int cls_id, bool is_person_model)
{
    Model m;
    m.net        = Darknet::load_neural_network(cfg, names, weights);
    m.conf_thresh = conf_thresh;
    m.cls_id      = cls_id;

    if (is_person_model)
    {
        // find "person" in the class names
        const auto & class_names = Darknet::get_class_names(m.net);
        for (int i = 0; i < (int)class_names.size(); ++i)
        {
            if (class_names[i] == "person") { m.person_class_idx = i; break; }
        }
    }

    return m;
}

// ---------------------------------------------------------------------------
// Infer one frame from all models, merge to vector<Detection>
// ---------------------------------------------------------------------------

static std::vector<Detection> infer_frame(
    const Model & person_model,
    const Model * cart_model,
    const cv::Mat & frame,
    track_kalman_t & tracker)
{
    std::vector<bbox_t> all_boxes;

    // Person detections
    {
        auto preds = Darknet::predict(person_model.net, frame);
        auto boxes = predictions_to_bbox(preds, CLS_PERSON, person_model.conf_thresh, person_model.person_class_idx);
        all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
    }

    // Cart detections (optional)
    if (cart_model)
    {
        auto preds = Darknet::predict(cart_model->net, frame);
        auto boxes = predictions_to_bbox(preds, CLS_CART, cart_model->conf_thresh, -1);
        all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
    }

    // Kalman tracker — returns same boxes with track_id assigned
    const auto tracked = tracker.correct(all_boxes);

    std::vector<Detection> dets;
    dets.reserve(tracked.size());
    for (const auto & b : tracked)
    {
        Detection d;
        d.rect     = cv::Rect(b.x, b.y, b.w, b.h);
        d.conf     = b.prob;
        d.cls_id   = b.obj_id;
        d.track_id = b.track_id;
        dets.push_back(d);
    }
    return dets;
}

// ---------------------------------------------------------------------------
// Console output
// ---------------------------------------------------------------------------

static void print_header()
{
    std::cout << "\n"
              << std::left  << std::setw(10) << "Role"
              << std::left  << std::setw(8)  << "ID"
              << std::left  << std::setw(10) << "Class"
              << std::right << std::setw(6)  << "Conf"
              << std::right << std::setw(11) << "Distance"
              << std::right << std::setw(9)  << "Angle"
              << "\n" << std::string(60, '-') << "\n";
}

static void print_row(const FrameResult & r, int frame_idx = -1)
{
    std::cout
        << std::left  << std::setw(10) << r.role
        << std::left  << std::setw(8)  << r.label_id
        << std::left  << std::setw(10) << r.class_tag
        << std::right << std::setw(5)  << static_cast<int>(r.conf * 100) << "%"
        << std::right << std::setw(9)  << std::fixed << std::setprecision(1) << r.dist << "m"
        << std::right << std::setw(8)  << std::showpos << std::setprecision(1) << r.angle << "°";
    if (frame_idx >= 0)
        std::cout << "  [f" << std::noshowpos << frame_idx << "]";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Run on a single image
// ---------------------------------------------------------------------------

static void run_image(const fs::path & path, const Model & person_model, const Model * cart_model)
{
    cv::Mat frame = cv::imread(path.string());
    if (frame.empty()) { std::cerr << "Could not load: " << path << "\n"; return; }

    track_kalman_t tracker(1000, 1, 80.0f, frame.size());
    const auto dets       = infer_frame(person_model, cart_model, frame, tracker);
    const int  target_idx = find_target_idx(dets, frame.cols, frame.rows);
    std::map<unsigned int, float> smooth_state;
    const auto rows       = annotate_frame(frame, dets, smooth_state, target_idx);

    std::cout << "\n" << path.filename().string()
              << " — " << rows.size() << " object(s) detected\n";
    print_header();
    for (const auto & r : rows) print_row(r);

    fs::path out_path = path.parent_path() / (path.stem().string() + "_tracked" + path.extension().string());
    cv::imwrite(out_path.string(), frame);
    std::cout << "\nSaved: " << out_path << "\n";
}

// ---------------------------------------------------------------------------
// Run on video / webcam
// ---------------------------------------------------------------------------

static void run_video(const std::string & source_str, const Model & person_model, const Model * cart_model)
{
    cv::VideoCapture cap;
    bool is_file = false;

    // Try parsing as integer (webcam index)
    try
    {
        int idx = std::stoi(source_str);
        cap.open(idx);
    }
    catch (...)
    {
        cap.open(source_str);
        is_file = true;
    }

    if (!cap.isOpened()) { std::cerr << "Could not open: " << source_str << "\n"; return; }

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int    w   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    h   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Kalman tracker: max_dist scales with frame resolution
    const float max_dist = std::min(w, h) * 0.12f;
    track_kalman_t tracker(1000, 3, max_dist, cv::Size(w, h));

    cv::VideoWriter writer;
    if (is_file)
    {
        fs::path p(source_str);
        fs::path out_path = p.parent_path() / (p.stem().string() + "_tracked" + p.extension().string());
        writer.open(out_path.string(), cv::VideoWriter::fourcc('m','p','4','v'),
                    fps > 0 ? fps : 30.0, cv::Size(w, h));
    }

    std::map<unsigned int, float> smooth_state;
    int frame_idx = 0;

    std::cout << "\nTracking — press Q to quit\n";
    print_header();

    cv::Mat frame;
    while (cap.read(frame))
    {
        if (frame.empty()) break;

        const auto dets       = infer_frame(person_model, cart_model, frame, tracker);
        const int  target_idx = find_target_idx(dets, frame.cols, frame.rows);
        const auto rows       = annotate_frame(frame, dets, smooth_state, target_idx);

        for (const auto & r : rows) print_row(r, frame_idx);

        cv::imshow("Cart Tracker", frame);
        cv::imshow("Overhead Map", draw_map(dets, target_idx, frame.cols, frame.rows));

        if (writer.isOpened()) writer.write(frame);

        if ((cv::waitKey(1) & 0xFF) == 'q') break;
        ++frame_idx;
    }

    cap.release();
    if (writer.isOpened())
    {
        writer.release();
        fs::path p(source_str);
        std::cout << "\nSaved: " << p.stem().string() << "_tracked" << p.extension().string() << "\n";
    }
    cv::destroyAllWindows();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image|video|webcam_id>\n";
        return 1;
    }

    const fs::path model_dir = fs::path(argv[0]).parent_path();

    const fs::path person_cfg     = model_dir / "yolov4-tiny.cfg";
    const fs::path person_weights = model_dir / "yolov4-tiny.weights";
    const fs::path person_names   = model_dir / "coco.names";

    for (const auto & p : {person_cfg, person_weights, person_names})
    {
        if (!fs::exists(p)) { std::cerr << "Missing: " << p << "\n"; return 1; }
    }

    Model person_model = load_model(person_cfg, person_names, person_weights,
                                    PERSON_CONF, CLS_PERSON, true);
    std::cout << "Loaded person model: " << person_weights.filename() << "\n";

    const fs::path cart_cfg     = model_dir / "cart_darknet.cfg";
    const fs::path cart_weights = model_dir / "cart_darknet.weights";
    const fs::path cart_names   = model_dir / "cart.names";

    Model cart_model_storage;
    Model * cart_model = nullptr;

    if (fs::exists(cart_cfg) && fs::exists(cart_weights) && fs::exists(cart_names))
    {
        cart_model_storage = load_model(cart_cfg, cart_names, cart_weights,
                                        CART_CONF, CLS_CART, false);
        cart_model = &cart_model_storage;
        std::cout << "Loaded cart model: " << cart_weights.filename() << "\n";
    }
    else
    {
        std::cout << "[warn] No darknet cart model — person detection only.\n";
    }

    const std::string source = argv[1];
    const bool is_image = [&]()
    {
        const std::string ext = fs::path(source).extension().string();
        for (const auto & e : {".jpg",".jpeg",".png",".bmp",".tiff",".webp"})
            if (ext == e) return true;
        return false;
    }();

    if (is_image)
        run_image(source, person_model, cart_model);
    else
        run_video(source, person_model, cart_model);

    Darknet::free_neural_network(person_model.net);
    if (cart_model) Darknet::free_neural_network(cart_model->net);

    return 0;
}
