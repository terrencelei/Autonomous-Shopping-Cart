// Constant two-motor speed test for the ESP32 cart motor controller.
//
// Pin layout:
//   M1 / right motor: PWM pins 18 and 19, encoder pins 32 and 33
//   M2 / left motor:  PWM pins 22 and 23, encoder pins 25 and 26
//
// Upload this sketch and both motors will run continuously at MOTOR_DUTY.
// Set MOTOR_DUTY lower if the cart is on the floor. Use Serial Monitor at
// 115200 baud to watch encoder counts.

// Motor 1 / right side.
#define M1_PWM_A 18
#define M1_PWM_B 19
#define M1_ENC_A 32
#define M1_ENC_B 33

// Motor 2 / left side.
#define M2_PWM_A 22
#define M2_PWM_B 23
#define M2_ENC_A 25
#define M2_ENC_B 26

#define PWM_FREQ 1000
#define PWM_RES  8

// 0 = stopped, 255 = full speed. Start conservatively.
const int MOTOR_DUTY = 80;

// Change either one to -1 if that motor needs to spin the opposite direction.
const int M1_DIRECTION = 1;
const int M2_DIRECTION = 1;

volatile long m1EncoderCount = 0;
volatile long m2EncoderCount = 0;
volatile uint8_t m1EncoderState = 0;
volatile uint8_t m2EncoderState = 0;

unsigned long lastPrintMs = 0;
long lastM1Count = 0;
long lastM2Count = 0;

int8_t IRAM_ATTR quadratureDelta(uint8_t previous, uint8_t current) {
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

uint8_t IRAM_ATTR readM1EncoderState() {
  return (digitalRead(M1_ENC_A) << 1) | digitalRead(M1_ENC_B);
}

uint8_t IRAM_ATTR readM2EncoderState() {
  return (digitalRead(M2_ENC_A) << 1) | digitalRead(M2_ENC_B);
}

void IRAM_ATTR m1EncoderISR() {
  uint8_t current = readM1EncoderState();
  m1EncoderCount += quadratureDelta(m1EncoderState, current);
  m1EncoderState = current;
}

void IRAM_ATTR m2EncoderISR() {
  uint8_t current = readM2EncoderState();
  m2EncoderCount += quadratureDelta(m2EncoderState, current);
  m2EncoderState = current;
}

void setMotor(int pwmA, int pwmB, int duty, int direction) {
  duty = constrain(duty, 0, 255);

  if (direction >= 0) {
    ledcWrite(pwmA, duty);
    ledcWrite(pwmB, 0);
  } else {
    ledcWrite(pwmA, 0);
    ledcWrite(pwmB, duty);
  }
}

void stopMotors() {
  ledcWrite(M1_PWM_A, 0);
  ledcWrite(M1_PWM_B, 0);
  ledcWrite(M2_PWM_A, 0);
  ledcWrite(M2_PWM_B, 0);
}

void runMotors() {
  setMotor(M1_PWM_A, M1_PWM_B, MOTOR_DUTY, M1_DIRECTION);
  setMotor(M2_PWM_A, M2_PWM_B, MOTOR_DUTY, M2_DIRECTION);
}

void setup() {
  Serial.begin(115200);

  ledcAttach(M1_PWM_A, PWM_FREQ, PWM_RES);
  ledcAttach(M1_PWM_B, PWM_FREQ, PWM_RES);
  ledcAttach(M2_PWM_A, PWM_FREQ, PWM_RES);
  ledcAttach(M2_PWM_B, PWM_FREQ, PWM_RES);

  stopMotors();

  pinMode(M1_ENC_A, INPUT_PULLUP);
  pinMode(M1_ENC_B, INPUT_PULLUP);
  pinMode(M2_ENC_A, INPUT_PULLUP);
  pinMode(M2_ENC_B, INPUT_PULLUP);

  m1EncoderState = readM1EncoderState();
  m2EncoderState = readM2EncoderState();

  attachInterrupt(digitalPinToInterrupt(M1_ENC_A), m1EncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(M1_ENC_B), m1EncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(M2_ENC_A), m2EncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(M2_ENC_B), m2EncoderISR, CHANGE);

  Serial.println("Constant motor speed test starting.");
  Serial.print("PWM duty: ");
  Serial.println(MOTOR_DUTY);

  delay(500);
  runMotors();
}

void loop() {
  runMotors();

  unsigned long now = millis();
  if (now - lastPrintMs >= 100) {
    lastPrintMs = now;

    noInterrupts();
    long m1Count = m1EncoderCount;
    long m2Count = m2EncoderCount;
    interrupts();

    long m1Delta = m1Count - lastM1Count;
    long m2Delta = m2Count - lastM2Count;
    lastM1Count = m1Count;
    lastM2Count = m2Count;

    Serial.print("M1 ");
    Serial.print(m1Count);
    Serial.print(" d ");
    Serial.print(m1Delta);
    Serial.print("    M2 ");
    Serial.print(m2Count);
    Serial.print(" d ");
    Serial.println(m2Delta);
  }
}
