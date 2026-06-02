// Autonomous Shopping Cart — ESP32 motor controller.
//
// Drives two TB9051FTG-controlled brushed DC motors and reports wheel
// encoder counts back over USB serial.  The matching host code is
// Pathfinding_algorithm.py / MotorDriver.send_velocities() running on
// the Raspberry Pi.
//
// ── Serial protocol (115200 baud) ──────────────────────────────
//   Host  → ESP32 :  "L<rpm> R<rpm>\n"   e.g.  "L42.5 R-30.0\n"
//                                        Target wheel RPM for each side.
//   ESP32 → Host  :  "E,<left_ticks>,<right_ticks>\n"
//                                        Cumulative encoder counts, sent
//                                        every REPORT_INTERVAL_MS.
//
// ── Safety ─────────────────────────────────────────────────────
//   If no command arrives within WATCHDOG_MS, the motors are stopped.
//   Unplugging the USB cable or crashing the Pi therefore cannot leave
//   the cart driving away.
//
// ── Pin layout (matches dual_motor_test.ino) ───────────────────
//   M1 (RIGHT) : PWM 18 / 19, encoder A=32 B=33
//   M2 (LEFT): PWM 22 / 23, encoder A=25 B=26
//
// ── Calibration ────────────────────────────────────────────────
//   MAX_RPM below is the wheel-shaft RPM that corresponds to a full-
//   scale setOutput(1.0).  Measure it by running the cart at 100 % for
//   a few seconds, counting encoder ticks, and converting:
//       rpm = (ticks_per_sec / ENCODER_PPR) * 60
//   Update the constant once measured.

struct WheelPid;

#include <TB9051FTGMotorCarrier.h>
#include <math.h>

// ── Pin definitions ───────────────────────────────────────────
#define RIGHT_PWM1   18
#define RIGHT_PWM2   19
#define RIGHT_ENC_A  32
#define RIGHT_ENC_B  33

#define LEFT_PWM1  22
#define LEFT_PWM2  23
#define LEFT_ENC_A 25
#define LEFT_ENC_B 26

// ── Drivers & encoder state ───────────────────────────────────
TB9051FTGMotorCarrier rightMotor(RIGHT_PWM1, RIGHT_PWM2);
TB9051FTGMotorCarrier leftMotor(LEFT_PWM1, LEFT_PWM2);

volatile long rightCount  = 0;
volatile long leftCount = 0;

volatile uint8_t rightEncState  = 0;
volatile uint8_t leftEncState = 0;

// Flip a sign if "forward" motion produces a negative count for that wheel.
#define RIGHT_ENC_DIR   1
#define LEFT_ENC_DIR  1

int8_t IRAM_ATTR quadratureDelta(uint8_t previous, uint8_t current) {
  // Two-bit (A << 1 | B) state machine; valid forward / reverse transitions
  // return +1 / -1, anything else (no movement, noisy double-edge) returns 0.
  switch ((previous << 2) | current) {
    case 0b0001:
    case 0b0111:
    case 0b1110:
    case 0b1000:
      return 1;
    case 0b0010:
    case 0b1011:
    case 0b1101:
    case 0b0100:
      return -1;
    default:
      return 0;
  }
}

uint8_t IRAM_ATTR readRightEncoderState() {
  return (digitalRead(RIGHT_ENC_A) << 1) | digitalRead(RIGHT_ENC_B);
}

uint8_t IRAM_ATTR readLeftEncoderState() {
  return (digitalRead(LEFT_ENC_A) << 1) | digitalRead(LEFT_ENC_B);
}

void IRAM_ATTR rightISR() {
  uint8_t cur = readRightEncoderState();
  rightCount += RIGHT_ENC_DIR * quadratureDelta(rightEncState, cur);
  rightEncState = cur;
}

void IRAM_ATTR leftISR() {
  uint8_t cur = readLeftEncoderState();
  leftCount += LEFT_ENC_DIR * quadratureDelta(leftEncState, cur);
  leftEncState = cur;
}

// ── Tunables ──────────────────────────────────────────────────
const float MAX_RPM = 100.0f;                 // calibrate to your hardware
const float ENCODER_PPR = 447.0f;             // must match Pathfinding_algorithm.py
const float RIGHT_RPM_SIGN = 1.0f;            // flip if measured RPM sign is reversed
const float LEFT_RPM_SIGN = 1.0f;

const unsigned long CONTROL_INTERVAL_MS = 20; // 50 Hz wheel-speed PID
const unsigned long WATCHDOG_MS        = 500;
const unsigned long REPORT_INTERVAL_MS = 20;  // 50 Hz encoder report

unsigned long lastCmdMs    = 0;
unsigned long lastControlMs = 0;
unsigned long lastReportMs = 0;
String        buf;

struct WheelPid {
  float targetRPM = 0.0f;
  float measuredRPM = 0.0f;
  float integral = 0.0f;
  float output = 0.0f;
  long lastCount = 0;
};

WheelPid rightPid;
WheelPid leftPid;

const float RPM_KP = 0.006f;
const float RPM_KI = 0.020f;
const float INTEGRAL_LIMIT = 25.0f;
const float STOP_RPM_EPS = 0.5f;

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void readEncoderCounts(long &right, long &left) {
  noInterrupts();
  right = rightCount;
  left = leftCount;
  interrupts();
}

float updatePid(WheelPid &pid, float measuredRPM, float dt) {
  pid.measuredRPM = measuredRPM;

  if (fabs(pid.targetRPM) < STOP_RPM_EPS) {
    pid.integral = 0.0f;
    pid.output = 0.0f;
    return 0.0f;
  }

  float error = pid.targetRPM - measuredRPM;
  pid.integral = clampf(pid.integral + error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

  float feedforward = pid.targetRPM / MAX_RPM;
  pid.output = clampf(feedforward
                      + RPM_KP * error
                      + RPM_KI * pid.integral,
                      -1.0f, 1.0f);
  return pid.output;
}

void setRightRPM(float rpm)  {
  rightPid.targetRPM = clampf(rpm, -MAX_RPM, MAX_RPM);
}
void setLeftRPM(float rpm) {
  leftPid.targetRPM = clampf(rpm, -MAX_RPM, MAX_RPM);
}
void stopMotors() {
  rightPid.targetRPM = 0.0f;
  leftPid.targetRPM = 0.0f;
  rightPid.integral = 0.0f;
  leftPid.integral = 0.0f;
  rightPid.output = 0.0f;
  leftPid.output = 0.0f;
  rightMotor.setOutput(0.0f);
  leftMotor.setOutput(0.0f);
}

void updateMotorPid(unsigned long nowMs) {
  if (nowMs - lastControlMs < CONTROL_INTERVAL_MS) return;

  float dt = (nowMs - lastControlMs) / 1000.0f;
  lastControlMs = nowMs;
  if (dt <= 0.0f) return;

  long rightNow, leftNow;
  readEncoderCounts(rightNow, leftNow);

  long rightDelta = rightNow - rightPid.lastCount;
  long leftDelta = leftNow - leftPid.lastCount;
  rightPid.lastCount = rightNow;
  leftPid.lastCount = leftNow;

  float rightMeasured = RIGHT_RPM_SIGN * (rightDelta / ENCODER_PPR) * (60.0f / dt);
  float leftMeasured = LEFT_RPM_SIGN * (leftDelta / ENCODER_PPR) * (60.0f / dt);

  rightMotor.setOutput(updatePid(rightPid, rightMeasured, dt));
  leftMotor.setOutput(updatePid(leftPid, leftMeasured, dt));
}

// Parse "L<rpm> R<rpm>" — tolerates extra whitespace.
void handleLine(const String &line) {
  int li = line.indexOf('L');
  int ri = line.indexOf('R');
  if (li < 0 || ri <= li) return;

  String lStr = line.substring(li + 1, ri); lStr.trim();
  String rStr = line.substring(ri + 1);     rStr.trim();
  if (lStr.length() == 0 || rStr.length() == 0) return;

  setLeftRPM(lStr.toFloat());
  setRightRPM(rStr.toFloat());
  lastCmdMs = millis();
}

// ── Setup / loop ──────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  rightMotor.enable();
  leftMotor.enable();

  // INPUT_PULLUP — not plain INPUT — so a disconnected or open-drain
  // encoder doesn't pick up PWM noise from neighbouring motor pins as
  // phantom edges. Replace with INPUT if you're using a push-pull
  // (totem-pole) encoder with its own external pull-down.
  pinMode(RIGHT_ENC_A,  INPUT_PULLUP);
  pinMode(RIGHT_ENC_B,  INPUT_PULLUP);
  pinMode(LEFT_ENC_A, INPUT_PULLUP);
  pinMode(LEFT_ENC_B, INPUT_PULLUP);

  // Seed the quadrature state machines with the current pin levels, then
  // attach CHANGE-edge ISRs on all four lines for full 4x decoding.
  rightEncState  = readRightEncoderState();
  leftEncState = readLeftEncoderState();
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A),  rightISR,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_B),  rightISR,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A), leftISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_B), leftISR, CHANGE);

  readEncoderCounts(rightPid.lastCount, leftPid.lastCount);
  lastControlMs = millis();
  stopMotors();
  Serial.println("cart_motor ready");
}

void loop() {
  // 1. Consume serial input, dispatch on newline.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      handleLine(buf);
      buf = "";
    } else if (c >= 32 && c < 127) {
      buf += c;
      if (buf.length() > 64) buf = "";  // guard against runaway input
    }
  }

  // 2. Watchdog — stop motors if the Pi has gone silent.
  if (millis() - lastCmdMs > WATCHDOG_MS) {
    stopMotors();
  }

  // 3. Closed-loop wheel-speed control.
  unsigned long nowMs = millis();
  updateMotorPid(nowMs);

  // 4. Periodic encoder report.
  if (nowMs - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = nowMs;
    long right, left;
    readEncoderCounts(right, left);
    Serial.print("E,");
    Serial.print(left);
    Serial.print(",");
    Serial.println(right);
  }
}
