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
//   M1 (LEFT) : PWM 18 / 19, encoder A=32 B=33
//   M2 (RIGHT): PWM 22 / 23, encoder A=25 B=26
//
// ── Calibration ────────────────────────────────────────────────
//   MAX_RPM below is the wheel-shaft RPM that corresponds to a full-
//   scale setOutput(1.0).  Measure it by running the cart at 100 % for
//   a few seconds, counting encoder ticks, and converting:
//       rpm = (ticks_per_sec / ENCODER_PPR) * 60
//   Update the constant once measured.

#include <TB9051FTGMotorCarrier.h>

// ── Pin definitions ───────────────────────────────────────────
#define LEFT_PWM1   18
#define LEFT_PWM2   19
#define LEFT_ENC_A  32
#define LEFT_ENC_B  33

#define RIGHT_PWM1  22
#define RIGHT_PWM2  23
#define RIGHT_ENC_A 25
#define RIGHT_ENC_B 26

// ── Drivers & encoder state ───────────────────────────────────
TB9051FTGMotorCarrier leftMotor(LEFT_PWM1, LEFT_PWM2);
TB9051FTGMotorCarrier rightMotor(RIGHT_PWM1, RIGHT_PWM2);

volatile long leftCount  = 0;
volatile long rightCount = 0;

void IRAM_ATTR leftISR()  {
  leftCount  += (digitalRead(LEFT_ENC_B)  == HIGH) ?  1 : -1;
}
void IRAM_ATTR rightISR() {
  rightCount += (digitalRead(RIGHT_ENC_B) == HIGH) ?  1 : -1;
}

// ── Tunables ──────────────────────────────────────────────────
const float MAX_RPM = 150.0f;                 // calibrate to your hardware
const unsigned long WATCHDOG_MS        = 500;
const unsigned long REPORT_INTERVAL_MS = 50;  // 20 Hz encoder report

unsigned long lastCmdMs    = 0;
unsigned long lastReportMs = 0;
String        buf;

// ── Motor helpers ─────────────────────────────────────────────
void setLeftRPM(float rpm)  {
  float u = rpm / MAX_RPM;
  if (u >  1.0f) u =  1.0f;
  if (u < -1.0f) u = -1.0f;
  leftMotor.setOutput(u);
}
void setRightRPM(float rpm) {
  float u = rpm / MAX_RPM;
  if (u >  1.0f) u =  1.0f;
  if (u < -1.0f) u = -1.0f;
  rightMotor.setOutput(u);
}
void stopMotors() {
  leftMotor.setOutput(0.0f);
  rightMotor.setOutput(0.0f);
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

  leftMotor.enable();
  rightMotor.enable();

  pinMode(LEFT_ENC_A,  INPUT);
  pinMode(LEFT_ENC_B,  INPUT);
  pinMode(RIGHT_ENC_A, INPUT);
  pinMode(RIGHT_ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A),  leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A), rightISR, RISING);

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

  // 3. Periodic encoder report.
  if (millis() - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = millis();
    noInterrupts();
    long l = leftCount;
    long r = rightCount;
    interrupts();
    Serial.print("E,");
    Serial.print(l);
    Serial.print(",");
    Serial.println(r);
  }
}
