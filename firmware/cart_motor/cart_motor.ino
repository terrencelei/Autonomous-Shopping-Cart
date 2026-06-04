// Autonomous Shopping Cart — ESP32 motor controller.
//
// Drives two TB9051FTG-controlled brushed DC motors open-loop and reports
// wheel encoder counts back over USB serial.  The matching host code is
// follow.py / MotorIO on the Raspberry Pi.
//
// ── Serial protocol (115200 baud) ──────────────────────────────
//   Host  → ESP32 :  "L<rpm> R<rpm>\n"   e.g.  "L42.5 R-30.0\n"
//                                        Open-loop speed command for each side.
//                                        MAX_RPM maps to full PWM output.
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
//   This sketch does not use encoders for feedback. MAX_RPM only scales
//   incoming L/R commands into PWM duty: output = command / MAX_RPM.

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

// ── Direct ESP32 PWM output & encoder state ───────────────────
const int PWM_FREQ = 1000;
const int PWM_RES = 8;
const int PWM_MAX_DUTY = 255;

// The motors are mounted as mirror images. The constant-speed test showed
// identical electrical polarity spins the two wheels in opposite directions,
// so invert the left electrical drive for matching positive wheel RPM.
const int RIGHT_MOTOR_DIR = 1;
const int LEFT_MOTOR_DIR = -1;

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
const unsigned long CONTROL_INTERVAL_MS = 20; // 50 Hz open-loop command refresh
const unsigned long WATCHDOG_MS        = 500;
const unsigned long REPORT_INTERVAL_MS = 20;  // 50 Hz encoder report

unsigned long lastCmdMs    = 0;
unsigned long lastControlMs = 0;
unsigned long lastReportMs = 0;
String        buf;

float rightTargetRPM = 0.0f;
float leftTargetRPM = 0.0f;

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

void setRightRPM(float rpm)  {
  rightTargetRPM = clampf(rpm, -MAX_RPM, MAX_RPM);
}
void setLeftRPM(float rpm) {
  leftTargetRPM = clampf(rpm, -MAX_RPM, MAX_RPM);
}

void setMotorOutput(int pwm1, int pwm2, float output, int motorDir) {
  output = clampf(output, -1.0f, 1.0f) * motorDir;
  int duty = (int)(fabs(output) * PWM_MAX_DUTY);

  if (output > 0.0f) {
    ledcWrite(pwm1, duty);
    ledcWrite(pwm2, 0);
  } else if (output < 0.0f) {
    ledcWrite(pwm1, 0);
    ledcWrite(pwm2, duty);
  } else {
    ledcWrite(pwm1, 0);
    ledcWrite(pwm2, 0);
  }
}

void setRightOutput(float output) {
  setMotorOutput(RIGHT_PWM1, RIGHT_PWM2, output, RIGHT_MOTOR_DIR);
}

void setLeftOutput(float output) {
  setMotorOutput(LEFT_PWM1, LEFT_PWM2, output, LEFT_MOTOR_DIR);
}

void stopMotors() {
  rightTargetRPM = 0.0f;
  leftTargetRPM = 0.0f;
  setRightOutput(0.0f);
  setLeftOutput(0.0f);
}

void updateMotorOutputs(unsigned long nowMs) {
  if (nowMs - lastControlMs < CONTROL_INTERVAL_MS) return;
  lastControlMs = nowMs;

  setRightOutput(rightTargetRPM / MAX_RPM);
  setLeftOutput(leftTargetRPM / MAX_RPM);
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

  ledcAttach(RIGHT_PWM1, PWM_FREQ, PWM_RES);
  ledcAttach(RIGHT_PWM2, PWM_FREQ, PWM_RES);
  ledcAttach(LEFT_PWM1, PWM_FREQ, PWM_RES);
  ledcAttach(LEFT_PWM2, PWM_FREQ, PWM_RES);

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

  lastControlMs = millis();
  stopMotors();
  Serial.println("cart_motor ready v5  open-loop direct-ledc  MOTOR_DIR L=-1 R=+1");
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

  // 3. Open-loop motor output.
  unsigned long nowMs = millis();
  updateMotorOutputs(nowMs);

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
