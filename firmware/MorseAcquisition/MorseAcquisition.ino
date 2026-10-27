/*
  Morse-vibration integrated hardware firmware
  ------------------------------------------------------------
  Target: ESP32-S3 N16R8 using the Arduino IDE
  Receiver: LSM6DS3 six-axis IMU over I2C
  Actuator: LRA driven by DRV2605L over I2C

  This sketch is the hardware-targeted version of the project firmware. It
  contains no synthetic sensor data and requires no third-party Arduino
  libraries. Arduino.h, Wire.h and the ESP32 Arduino core are sufficient.

  IMPORTANT: values in the USER CONFIGURATION section marked VERIFY must be
  checked against the exact breakout boards and LRA after delivery.

  Physical connections
  --------------------
  ESP32 3V3  -> LSM6DS3 VIN/3V3 (only if the breakout accepts 3.3 V)
  ESP32 GND  -> LSM6DS3 GND, DRV2605L GND, actuator-supply GND
  GPIO 8     -> LSM6DS3 SDA and DRV2605L SDA
  GPIO 9     -> LSM6DS3 SCL and DRV2605L SCL
  GPIO 4     -> Morse button; other button terminal to GND
  GPIO 5     -> Record button; other button terminal to GND
  DRV2605L VCC/VIN -> supply specified by the purchased module
  DRV2605L OUT+/-  -> LRA terminals

  Both buttons use INPUT_PULLUP. A saved key_state of 1 means pressed.
  Do not power the LRA from an ESP32 GPIO.

  Normal collection procedure
  ---------------------------
  1. Power the rig and confirm that both I2C devices are detected.
  2. Leave the rig undisturbed and send c, or hold RECORD for two seconds.
  3. Wait for the five-second calibration to finish and READY to appear.
  4. Send r or short-press RECORD to start a bounded recording.
  5. Enter Morse with the Morse button.
  6. Send r or short-press RECORD to stop.

  The same calibration remains valid for subsequent recordings. Recalibrate
  after changing the mounting, surface, sensor orientation or environment.

  Serial commands
  ---------------
  h : help                         i : status
  c : five-second calibration      r : start/stop recording
  1 : low LRA strength             2 : medium LRA strength
  3 : high LRA strength            t : readable 50 Hz text output
  b : full-rate binary output
  f : full collection mode         k : button test mode
  a : IMU test mode                l : LRA test mode
  s : synchronisation test mode

  Binary packet version 3 is 42 bytes, little-endian, and ends with a
  CRC-16/CCITT-FALSE. At 1660 samples/s it produces approximately 69.7 kB/s.
*/

#include <Arduino.h>
#include <Wire.h>
#include <esp_timer.h>
#include <math.h>
#include <stdlib.h>

// ======================== USER CONFIGURATION ============================

constexpr uint32_t SERIAL_BAUD = 921600;

// ESP32-S3 pins. VERIFY against the labels/pinout of the purchased board.
constexpr uint8_t PIN_MORSE_BUTTON = 4;
constexpr uint8_t PIN_RECORD_BUTTON = 5;
constexpr uint8_t PIN_I2C_SDA = 8;
constexpr uint8_t PIN_I2C_SCL = 9;

// Both devices share one 400 kHz I2C bus.
constexpr uint32_t I2C_CLOCK_HZ = 400000;
constexpr uint8_t DRV2605_ADDRESS = 0x5A;
constexpr uint8_t IMU_ADDRESS_PREFERENCE = 0x6A;  // Automatically tries 0x6B too.

// LSM6DS3 configuration. VERIFY using pilot data.
constexpr uint32_t IMU_OUTPUT_RATE_HZ = 833;  // Device setting is 1.66 kHz but i made it lower.
constexpr uint32_t NOMINAL_SAMPLE_PERIOD_US = 1000000UL / IMU_OUTPUT_RATE_HZ;
constexpr uint8_t LSM6DS3_CTRL1_XL_VALUE = 0x78; // 833 Hz, +/-4 g, 400 Hz BW.
constexpr uint8_t LSM6DS3_CTRL2_G_VALUE = 0x74;  // 833 Hz, +/-500 dps.

// DRV2605L/LRA starter values. VERIFY with the exact LRA datasheet.
// These two registers affect actuator voltage and must be checked before
// prolonged use. 0x50 and 0x89 are common starter values, not universal ones.
constexpr uint8_t DRV_RATED_VOLTAGE_REGISTER = 0x50;
constexpr uint8_t DRV_OVERDRIVE_CLAMP_REGISTER = 0x89;
constexpr uint8_t LRA_LIBRARY = 6;

// Unsigned real-time playback commands. Adjust after physical testing.
constexpr uint8_t LRA_STRENGTH_LOW = 64;
constexpr uint8_t LRA_STRENGTH_MEDIUM = 128;
constexpr uint8_t LRA_STRENGTH_HIGH = 192;

constexpr uint32_t CALIBRATION_DURATION_MS = 5000;
constexpr uint32_t RECORD_LONG_PRESS_MS = 2000;
constexpr uint32_t DEBOUNCE_MS = 20;
constexpr uint32_t TEXT_OUTPUT_RATE_HZ = 50;
constexpr uint32_t IMU_TEST_OUTPUT_RATE_HZ = 20;

// Packet buffering uses PSRAM when available and ordinary RAM otherwise.
constexpr size_t PSRAM_PACKET_CAPACITY = 4096;   // About 172 kB.
constexpr size_t RAM_PACKET_CAPACITY = 256;      // About 10.5 kB.

// ========================================================================

// LSM6DS3 registers used by this sketch.
constexpr uint8_t LSM_REG_WHO_AM_I = 0x0F;
constexpr uint8_t LSM_REG_CTRL1_XL = 0x10;
constexpr uint8_t LSM_REG_CTRL2_G = 0x11;
constexpr uint8_t LSM_REG_CTRL3_C = 0x12;
constexpr uint8_t LSM_REG_STATUS = 0x1E;
constexpr uint8_t LSM_REG_OUTX_L_G = 0x22;
constexpr uint8_t LSM_WHO_AM_I_VALUE = 0x69;

// DRV2605L registers used by this sketch.
constexpr uint8_t DRV_REG_STATUS = 0x00;
constexpr uint8_t DRV_REG_MODE = 0x01;
constexpr uint8_t DRV_REG_RTP_INPUT = 0x02;
constexpr uint8_t DRV_REG_LIBRARY = 0x03;
constexpr uint8_t DRV_REG_RATED_VOLTAGE = 0x16;
constexpr uint8_t DRV_REG_OVERDRIVE_CLAMP = 0x17;
constexpr uint8_t DRV_REG_FEEDBACK_CONTROL = 0x1A;
constexpr uint8_t DRV_REG_CONTROL3 = 0x1D;
constexpr uint8_t DRV_MODE_RTP = 0x05;

constexpr uint16_t PACKET_MAGIC = 0xA55A;
constexpr uint8_t PACKET_VERSION = 3;
constexpr uint8_t PACKET_TYPE_SAMPLE = 1;

enum class SystemState : uint8_t {
  UNCALIBRATED = 0,
  CALIBRATING = 1,
  READY = 2,
  RECORDING = 3
};

enum class OutputMode : uint8_t {
  TEXT_MONITOR = 0,
  BINARY_FULL_RATE = 1
};

enum class OperatingMode : uint8_t {
  FULL_COLLECTION = 0,
  BUTTON_TEST = 1,
  IMU_TEST = 2,
  LRA_TEST = 3,
  SYNC_TEST = 4
};

struct ImuRawSample {
  int16_t accelX;
  int16_t accelY;
  int16_t accelZ;
  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;
};

// Keep this definition matched to the laptop logger.
struct __attribute__((packed)) SamplePacket {
  uint16_t magic;
  uint8_t version;
  uint8_t packetType;
  uint32_t recordingId;
  uint32_t calibrationId;
  uint32_t sampleIndex;
  uint64_t timestampUs;
  int16_t accelX;
  int16_t accelY;
  int16_t accelZ;
  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;
  uint8_t keyState;
  uint8_t motorCommand;
  uint8_t systemState;
  uint8_t statusFlags;
  uint16_t crc16;
};

static_assert(sizeof(SamplePacket) == 42, "SamplePacket must be 42 bytes");

class DebouncedButton {
 public:
  explicit DebouncedButton(uint8_t pin) : pin_(pin) {}

  void begin() {
    pinMode(pin_, INPUT_PULLUP);
    rawPressed_ = digitalRead(pin_) == LOW;
    stablePressed_ = rawPressed_;
    lastRawChangeMs_ = millis();
  }

  void update(uint32_t nowMs) {
    pressedEdge_ = false;
    releasedEdge_ = false;
    const bool rawNow = digitalRead(pin_) == LOW;

    if (rawNow != rawPressed_) {
      rawPressed_ = rawNow;
      lastRawChangeMs_ = nowMs;
    }

    if (rawPressed_ != stablePressed_ &&
        static_cast<uint32_t>(nowMs - lastRawChangeMs_) >= DEBOUNCE_MS) {
      stablePressed_ = rawPressed_;
      pressedEdge_ = stablePressed_;
      releasedEdge_ = !stablePressed_;
    }
  }

  bool isPressed() const { return stablePressed_; }
  bool wasPressed() const { return pressedEdge_; }
  bool wasReleased() const { return releasedEdge_; }

 private:
  uint8_t pin_;
  bool rawPressed_ = false;
  bool stablePressed_ = false;
  bool pressedEdge_ = false;
  bool releasedEdge_ = false;
  uint32_t lastRawChangeMs_ = 0;
};

class RunningStats {
 public:
  void reset() {
    count_ = 0;
    mean_ = 0.0;
    m2_ = 0.0;
  }

  void add(double value) {
    ++count_;
    const double delta = value - mean_;
    mean_ += delta / static_cast<double>(count_);
    const double delta2 = value - mean_;
    m2_ += delta * delta2;
  }

  uint32_t count() const { return count_; }
  double mean() const { return mean_; }

  double standardDeviation() const {
    if (count_ < 2) return 0.0;
    return sqrt(m2_ / static_cast<double>(count_ - 1));
  }

 private:
  uint32_t count_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;
};

DebouncedButton morseButton(PIN_MORSE_BUTTON);
DebouncedButton recordButton(PIN_RECORD_BUTTON);

SystemState systemState = SystemState::UNCALIBRATED;
OutputMode outputMode = OutputMode::TEXT_MONITOR;
OperatingMode operatingMode = OperatingMode::FULL_COLLECTION;

bool imuReady = false;
bool drvReady = false;
uint8_t imuAddress = 0;
uint32_t i2cErrorCount = 0;

uint32_t calibrationId = 0;
uint32_t recordingId = 0;
uint32_t sampleIndex = 0;
uint64_t captureStartUs = 0;
uint64_t calibrationEndUs = 0;
uint64_t previousSampleUs = 0;

uint8_t selectedMotorStrength = LRA_STRENGTH_MEDIUM;
uint8_t currentMotorCommand = 0;

uint32_t timingOverrunCount = 0;
uint32_t imuNotReadyCount = 0;
uint32_t packetQueueOverflowCount = 0;
uint32_t packetsProduced = 0;

uint32_t recordButtonPressedAtMs = 0;
bool recordLongPressHandled = false;
bool previousTestMorseState = false;
bool previousTestRecordState = false;
uint64_t lastImuTestPrintUs = 0;

RunningStats accelXStats;
RunningStats accelYStats;
RunningStats accelZStats;
RunningStats gyroXStats;
RunningStats gyroYStats;
RunningStats gyroZStats;

SamplePacket* packetQueue = nullptr;
size_t packetQueueCapacity = 0;
size_t packetQueueHead = 0;
size_t packetQueueTail = 0;
size_t packetQueueCount = 0;

uint64_t nowMicroseconds() {
  return static_cast<uint64_t>(esp_timer_get_time());
}

const char* stateName(SystemState state) {
  switch (state) {
    case SystemState::UNCALIBRATED: return "UNCALIBRATED";
    case SystemState::CALIBRATING: return "CALIBRATING";
    case SystemState::READY: return "READY";
    case SystemState::RECORDING: return "RECORDING";
    default: return "UNKNOWN";
  }
}

const char* operatingModeName(OperatingMode mode) {
  switch (mode) {
    case OperatingMode::FULL_COLLECTION: return "FULL_COLLECTION";
    case OperatingMode::BUTTON_TEST: return "BUTTON_TEST";
    case OperatingMode::IMU_TEST: return "IMU_TEST";
    case OperatingMode::LRA_TEST: return "LRA_TEST";
    case OperatingMode::SYNC_TEST: return "SYNC_TEST";
    default: return "UNKNOWN";
  }
}

bool isCapturing() {
  return systemState == SystemState::CALIBRATING ||
         systemState == SystemState::RECORDING;
}

bool calibrationIsValid() {
  return calibrationId > 0 &&
         systemState != SystemState::UNCALIBRATED &&
         systemState != SystemState::CALIBRATING;
}

uint16_t crc16CcittFalse(const uint8_t* data, size_t length) {
  static constexpr uint16_t table[16] = {
    0x0000, 0x1021, 0x2042, 0x3063,
    0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B,
    0xC18C, 0xD1AD, 0xE1CE, 0xF1EF
  };

  uint16_t crc = 0xFFFF;

  for (size_t i = 0; i < length; ++i) {
    uint8_t index = ((crc >> 12) ^ (data[i] >> 4)) & 0x0F;
    crc = static_cast<uint16_t>((crc << 4) ^ table[index]);

    index = ((crc >> 12) ^ (data[i] & 0x0F)) & 0x0F;
    crc = static_cast<uint16_t>((crc << 4) ^ table[index]);
  }

  return crc;
}

bool i2cProbe(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

bool i2cWriteRegister(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  const uint8_t result = Wire.endTransmission();
  if (result != 0) ++i2cErrorCount;
  return result == 0;
}

bool i2cReadRegisters(uint8_t address, uint8_t startRegister,
                      uint8_t* destination, size_t length) {
  Wire.beginTransmission(address);
  Wire.write(startRegister);
  if (Wire.endTransmission(false) != 0) {
    ++i2cErrorCount;
    return false;
  }

  const size_t received = Wire.requestFrom(static_cast<int>(address),
                                            static_cast<int>(length));
  if (received != length) {
    ++i2cErrorCount;
    while (Wire.available()) Wire.read();
    return false;
  }

  for (size_t i = 0; i < length; ++i) {
    destination[i] = static_cast<uint8_t>(Wire.read());
  }
  return true;
}

bool i2cReadRegister(uint8_t address, uint8_t reg, uint8_t& value) {
  return i2cReadRegisters(address, reg, &value, 1);
}

bool i2cUpdateBits(uint8_t address, uint8_t reg,
                   uint8_t clearMask, uint8_t setMask) {
  uint8_t value = 0;
  if (!i2cReadRegister(address, reg, value)) return false;
  value = static_cast<uint8_t>((value & ~clearMask) | setMask);
  return i2cWriteRegister(address, reg, value);
}

bool beginImu() {
  const uint8_t alternateAddress =
      IMU_ADDRESS_PREFERENCE == 0x6A ? 0x6B : 0x6A;

  if (i2cProbe(IMU_ADDRESS_PREFERENCE)) {
    imuAddress = IMU_ADDRESS_PREFERENCE;
  } else if (i2cProbe(alternateAddress)) {
    imuAddress = alternateAddress;
  } else {
    Serial.println("# ERROR: LSM6DS3 not detected at 0x6A or 0x6B");
    return false;
  }

  uint8_t whoAmI = 0;
  if (!i2cReadRegister(imuAddress, LSM_REG_WHO_AM_I, whoAmI) ||
      whoAmI != LSM_WHO_AM_I_VALUE) {
    Serial.printf("# ERROR: unexpected IMU WHO_AM_I=0x%02X; expected 0x69\n",
                  whoAmI);
    return false;
  }

  // Software reset, then wait for the device to clear the reset bit.
  if (!i2cWriteRegister(imuAddress, LSM_REG_CTRL3_C, 0x01)) return false;
  const uint32_t resetStartedMs = millis();
  uint8_t ctrl3 = 0x01;
  while ((ctrl3 & 0x01) != 0 && millis() - resetStartedMs < 100) {
    delay(2);
    if (!i2cReadRegister(imuAddress, LSM_REG_CTRL3_C, ctrl3)) return false;
  }
  if ((ctrl3 & 0x01) != 0) {
    Serial.println("# ERROR: LSM6DS3 reset timeout");
    return false;
  }

  // BDU=1 prevents high/low bytes changing mid-read. IF_INC=1 enables burst.
  if (!i2cWriteRegister(imuAddress, LSM_REG_CTRL3_C, 0x44)) return false;
  if (!i2cWriteRegister(imuAddress, LSM_REG_CTRL1_XL,
                        LSM6DS3_CTRL1_XL_VALUE)) return false;
  if (!i2cWriteRegister(imuAddress, LSM_REG_CTRL2_G,
                        LSM6DS3_CTRL2_G_VALUE)) return false;
  delay(20);

  Serial.printf("# LSM6DS3 ready,address=0x%02X,odr_hz=%lu,accel=+/-4g,gyro=+/-500dps\n",
                imuAddress, static_cast<unsigned long>(IMU_OUTPUT_RATE_HZ));
  return true;
}

bool imuDataReady() {
  uint8_t status = 0;
  if (!i2cReadRegister(imuAddress, LSM_REG_STATUS, status)) return false;
  return (status & 0x03) == 0x03;  // Both accelerometer and gyro new-data bits.
}

int16_t makeInt16(uint8_t lowByte, uint8_t highByte) {
  return static_cast<int16_t>(static_cast<uint16_t>(lowByte) |
                              (static_cast<uint16_t>(highByte) << 8));
}

bool readImuRaw(ImuRawSample& sample) {
  uint8_t bytes[12];
  if (!i2cReadRegisters(imuAddress, LSM_REG_OUTX_L_G, bytes, sizeof(bytes))) {
    return false;
  }

  // LSM6DS3 burst order is gyro XYZ followed by accelerometer XYZ.
  sample.gyroX = makeInt16(bytes[0], bytes[1]);
  sample.gyroY = makeInt16(bytes[2], bytes[3]);
  sample.gyroZ = makeInt16(bytes[4], bytes[5]);
  sample.accelX = makeInt16(bytes[6], bytes[7]);
  sample.accelY = makeInt16(bytes[8], bytes[9]);
  sample.accelZ = makeInt16(bytes[10], bytes[11]);
  return true;
}

bool beginDriver() {
  if (!i2cProbe(DRV2605_ADDRESS)) {
    Serial.println("# ERROR: DRV2605L not detected at 0x5A");
    return false;
  }

  // Leave standby, stop output, and select the LRA waveform library.
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_MODE, 0x00)) return false;
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_RTP_INPUT, 0x00)) return false;
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_LIBRARY, LRA_LIBRARY)) return false;

  // VERIFY these voltage settings against the exact purchased LRA.
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_RATED_VOLTAGE,
                        DRV_RATED_VOLTAGE_REGISTER)) return false;
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_OVERDRIVE_CLAMP,
                        DRV_OVERDRIVE_CLAMP_REGISTER)) return false;

  // FEEDBACK_CONTROL bit 7: 1=LRA, 0=ERM.
  if (!i2cUpdateBits(DRV2605_ADDRESS, DRV_REG_FEEDBACK_CONTROL,
                     0x80, 0x80)) return false;
                    
  // Control2 register 0x1C:
  // Clear bit 7 (BIDIR_INPUT) for closed-loop unidirectional control.
  if (!i2cUpdateBits(DRV2605_ADDRESS, 0x1C, 0x80, 0x00)) {
  return false;
  }                   

  // CONTROL3 bit 3: 1=unsigned RTP input (0..255).
  if (!i2cUpdateBits(DRV2605_ADDRESS, DRV_REG_CONTROL3,
                     0x08, 0x08)) return false;

  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_MODE,
                        DRV_MODE_RTP)) return false;
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_RTP_INPUT, 0x00)) return false;

  uint8_t status = 0;
  i2cReadRegister(DRV2605_ADDRESS, DRV_REG_STATUS, status);
  Serial.printf("# DRV2605L ready,address=0x5A,status=0x%02X,mode=RTP,unsigned=1\n",
                status);
  Serial.println("# VERIFY LRA rated-voltage and overdrive-clamp register values before extended use");
  return true;
}

bool setMotorCommand(uint8_t command) {
  if (!drvReady) {
    currentMotorCommand = 0;
    return command == 0;
  }
  if (command == currentMotorCommand) return true;
  if (!i2cWriteRegister(DRV2605_ADDRESS, DRV_REG_RTP_INPUT, command)) {
    currentMotorCommand = 0;
    return false;
  }
  currentMotorCommand = command;
  return true;
}

void allocatePacketQueue() {
  packetQueueCapacity = 0;
  packetQueue = nullptr;

#if defined(ARDUINO_ARCH_ESP32)
  if (psramFound()) {
    packetQueue = static_cast<SamplePacket*>(
        ps_malloc(PSRAM_PACKET_CAPACITY * sizeof(SamplePacket)));
    if (packetQueue != nullptr) packetQueueCapacity = PSRAM_PACKET_CAPACITY;
  }
#endif

  if (packetQueue == nullptr) {
    packetQueue = static_cast<SamplePacket*>(
        malloc(RAM_PACKET_CAPACITY * sizeof(SamplePacket)));
    if (packetQueue != nullptr) packetQueueCapacity = RAM_PACKET_CAPACITY;
  }

  if (packetQueue == nullptr) {
    Serial.println("# WARNING: binary packet queue allocation failed");
  } else {
    Serial.printf("# packet_queue_capacity=%u,bytes=%u\n",
                  static_cast<unsigned>(packetQueueCapacity),
                  static_cast<unsigned>(packetQueueCapacity * sizeof(SamplePacket)));
  }
}

void clearPacketQueue() {
  packetQueueHead = 0;
  packetQueueTail = 0;
  packetQueueCount = 0;
}

bool enqueuePacket(const SamplePacket& packet) {
  if (packetQueue == nullptr || packetQueueCount >= packetQueueCapacity) {
    ++packetQueueOverflowCount;
    return false;
  }
  packetQueue[packetQueueHead] = packet;
  packetQueueHead = (packetQueueHead + 1) % packetQueueCapacity;
  ++packetQueueCount;
  return true;
}

void drainPacketQueue() {
  if (outputMode != OutputMode::BINARY_FULL_RATE || packetQueueCount == 0) return;

  while (packetQueueCount > 0 &&
         Serial.availableForWrite() >= static_cast<int>(sizeof(SamplePacket))) {
    Serial.write(reinterpret_cast<const uint8_t*>(&packetQueue[packetQueueTail]),
                 sizeof(SamplePacket));
    packetQueueTail = (packetQueueTail + 1) % packetQueueCapacity;
    --packetQueueCount;
  }
}

void resetCalibrationStatistics() {
  accelXStats.reset();
  accelYStats.reset();
  accelZStats.reset();
  gyroXStats.reset();
  gyroYStats.reset();
  gyroZStats.reset();
}

void addCalibrationSample(const ImuRawSample& sample) {
  accelXStats.add(sample.accelX);
  accelYStats.add(sample.accelY);
  accelZStats.add(sample.accelZ);
  gyroXStats.add(sample.gyroX);
  gyroYStats.add(sample.gyroY);
  gyroZStats.add(sample.gyroZ);
}

void printCalibrationStatistics() {
  if (outputMode != OutputMode::TEXT_MONITOR) return;
  Serial.println("# CALIBRATION_STATS,channel,mean_raw,std_raw");
  Serial.printf("# CALIBRATION_STATS,accel_x,%.3f,%.3f\n",
                accelXStats.mean(), accelXStats.standardDeviation());
  Serial.printf("# CALIBRATION_STATS,accel_y,%.3f,%.3f\n",
                accelYStats.mean(), accelYStats.standardDeviation());
  Serial.printf("# CALIBRATION_STATS,accel_z,%.3f,%.3f\n",
                accelZStats.mean(), accelZStats.standardDeviation());
  Serial.printf("# CALIBRATION_STATS,gyro_x,%.3f,%.3f\n",
                gyroXStats.mean(), gyroXStats.standardDeviation());
  Serial.printf("# CALIBRATION_STATS,gyro_y,%.3f,%.3f\n",
                gyroYStats.mean(), gyroYStats.standardDeviation());
  Serial.printf("# CALIBRATION_STATS,gyro_z,%.3f,%.3f\n",
                gyroZStats.mean(), gyroZStats.standardDeviation());
}

void printCsvHeader() {
  Serial.println("recording_id,calibration_id,sample_index,timestamp_us,"
                 "accel_x_raw,accel_y_raw,accel_z_raw,gyro_x_raw,gyro_y_raw,"
                 "gyro_z_raw,key_state,motor_command,state,status_flags");
}

void printHelp() {
  Serial.println();
  Serial.println("Morse-vibration integrated hardware firmware");
  Serial.println("  h help       i status       c calibrate       r start/stop");
  Serial.println("  1 low LRA    2 medium LRA   3 high LRA");
  Serial.println("  t text 50Hz  b binary full-rate");
  Serial.println("  f full mode  k button test  a IMU test");
  Serial.println("  l LRA test   s synchronisation test");
  Serial.println("Record button: hold 2 s to calibrate; short press to start/stop.");
  Serial.println("A calibration is reused until c is sent again or power is removed.");
  Serial.println();
}

void printStatus() {
  if (outputMode != OutputMode::TEXT_MONITOR) return;
  Serial.printf("# STATUS,mode=%s,state=%s,imu=%u,imu_address=0x%02X,drv=%u,"
                "calibration=%lu,recording=%lu,samples=%lu,overruns=%lu,"
                "imu_not_ready=%lu,i2c_errors=%lu,queue=%u,queue_overflows=%lu,"
                "strength=%u,output=text\n",
                operatingModeName(operatingMode), stateName(systemState),
                imuReady ? 1 : 0, imuAddress, drvReady ? 1 : 0,
                static_cast<unsigned long>(calibrationId),
                static_cast<unsigned long>(recordingId),
                static_cast<unsigned long>(sampleIndex),
                static_cast<unsigned long>(timingOverrunCount),
                static_cast<unsigned long>(imuNotReadyCount),
                static_cast<unsigned long>(i2cErrorCount),
                static_cast<unsigned>(packetQueueCount),
                static_cast<unsigned long>(packetQueueOverflowCount),
                selectedMotorStrength);
}

void resetCaptureCounters() {
  sampleIndex = 0;
  timingOverrunCount = 0;
  imuNotReadyCount = 0;
  packetQueueOverflowCount = 0;
  packetsProduced = 0;
  previousSampleUs = 0;
  clearPacketQueue();
}

void startCalibration() {
  if (!imuReady) {
    if (outputMode == OutputMode::TEXT_MONITOR)
      Serial.println("# CALIBRATION_REFUSED: IMU is not ready");
    return;
  }
  if (systemState == SystemState::RECORDING) {
    if (outputMode == OutputMode::TEXT_MONITOR)
      Serial.println("# CALIBRATION_REFUSED: stop recording first");
    return;
  }

  setMotorCommand(0);
  ++calibrationId;
  resetCaptureCounters();
  resetCalibrationStatistics();
  captureStartUs = nowMicroseconds();
  calibrationEndUs = captureStartUs +
      static_cast<uint64_t>(CALIBRATION_DURATION_MS) * 1000ULL;
  systemState = SystemState::CALIBRATING;

  if (outputMode == OutputMode::TEXT_MONITOR) {
    Serial.printf("# CALIBRATION_START,id=%lu,duration_ms=%lu,odr_hz=%lu\n",
                  static_cast<unsigned long>(calibrationId),
                  static_cast<unsigned long>(CALIBRATION_DURATION_MS),
                  static_cast<unsigned long>(IMU_OUTPUT_RATE_HZ));
    printCsvHeader();
  }
}

void finishCalibration() {
  if (systemState != SystemState::CALIBRATING) return;
  systemState = SystemState::READY;
  printCalibrationStatistics();
  if (outputMode == OutputMode::TEXT_MONITOR) {
    Serial.printf("# CALIBRATION_END,id=%lu,samples=%lu,overruns=%lu,"
                  "queue_overflows=%lu\n",
                  static_cast<unsigned long>(calibrationId),
                  static_cast<unsigned long>(sampleIndex),
                  static_cast<unsigned long>(timingOverrunCount),
                  static_cast<unsigned long>(packetQueueOverflowCount));
    Serial.println("# STATE=READY");
  }
}

void startRecording() {
  if (!imuReady || !drvReady) {
    if (outputMode == OutputMode::TEXT_MONITOR)
      Serial.println("# RECORDING_REFUSED: IMU and DRV2605L must both be ready");
    return;
  }

  const bool syncWithoutCalibration =
      operatingMode == OperatingMode::SYNC_TEST &&
      systemState == SystemState::UNCALIBRATED;
  if (systemState != SystemState::READY && !syncWithoutCalibration) {
    if (outputMode == OutputMode::TEXT_MONITOR)
      Serial.println("# RECORDING_REFUSED: perform calibration first");
    return;
  }

  ++recordingId;
  resetCaptureCounters();
  captureStartUs = nowMicroseconds();
  systemState = SystemState::RECORDING;
  setMotorCommand(0);

  if (outputMode == OutputMode::TEXT_MONITOR) {
    Serial.printf("# RECORDING_START,id=%lu,calibration_id=%lu,odr_hz=%lu,"
                  "strength=%u,mode=%s\n",
                  static_cast<unsigned long>(recordingId),
                  static_cast<unsigned long>(calibrationId),
                  static_cast<unsigned long>(IMU_OUTPUT_RATE_HZ),
                  selectedMotorStrength, operatingModeName(operatingMode));
    printCsvHeader();
  }
}

void stopRecording() {
  if (systemState != SystemState::RECORDING) return;
  setMotorCommand(0);
  systemState = calibrationId > 0 ? SystemState::READY
                                  : SystemState::UNCALIBRATED;

  if (outputMode == OutputMode::TEXT_MONITOR) {
    Serial.printf("# RECORDING_END,id=%lu,calibration_id=%lu,samples=%lu,"
                  "overruns=%lu,queue_overflows=%lu\n",
                  static_cast<unsigned long>(recordingId),
                  static_cast<unsigned long>(calibrationId),
                  static_cast<unsigned long>(sampleIndex),
                  static_cast<unsigned long>(timingOverrunCount),
                  static_cast<unsigned long>(packetQueueOverflowCount));
    Serial.printf("# STATE=%s\n", stateName(systemState));
  }
}

void toggleRecording() {
  if (systemState == SystemState::RECORDING) stopRecording();
  else startRecording();
}

bool effectiveKeyPressed() {
  if (operatingMode == OperatingMode::LRA_TEST) return morseButton.isPressed();
  if (systemState == SystemState::RECORDING) return morseButton.isPressed();
  return false;
}

void updateMotorFromKey() {
  const uint8_t requested = effectiveKeyPressed() ? selectedMotorStrength : 0;
  setMotorCommand(requested);
}

void setOperatingMode(OperatingMode newMode) {
  if (isCapturing()) {
    if (outputMode == OutputMode::TEXT_MONITOR)
      Serial.println("# Stop or finish capture before changing operating mode");
    return;
  }
  setMotorCommand(0);
  operatingMode = newMode;
  previousTestMorseState = morseButton.isPressed();
  previousTestRecordState = recordButton.isPressed();
  if (outputMode == OutputMode::TEXT_MONITOR)
    Serial.printf("# operating_mode=%s\n", operatingModeName(operatingMode));
}

void selectStrength(uint8_t strength, const char* label) {
  selectedMotorStrength = strength;
  updateMotorFromKey();
  if (outputMode == OutputMode::TEXT_MONITOR)
    Serial.printf("# strength=%s,command=%u\n", label, strength);
}

void processSerialCommand(char command) {
  switch (command) {
    case 'h': case 'H': printHelp(); break;
    case 'i': case 'I': printStatus(); break;
    case 'c': case 'C': startCalibration(); break;
    case 'r': case 'R': toggleRecording(); break;
    case '1': selectStrength(LRA_STRENGTH_LOW, "LOW"); break;
    case '2': selectStrength(LRA_STRENGTH_MEDIUM, "MEDIUM"); break;
    case '3': selectStrength(LRA_STRENGTH_HIGH, "HIGH"); break;

    case 't': case 'T':
      if (isCapturing() || packetQueueCount > 0) {
        if (outputMode == OutputMode::TEXT_MONITOR)
          Serial.println("# Wait for capture/queued binary data to finish before text mode");
      } else {
        outputMode = OutputMode::TEXT_MONITOR;
        Serial.println("# output=TEXT_MONITOR,display_rate_hz=50");
      }
      break;

    case 'b': case 'B':
      if (isCapturing()) {
        if (outputMode == OutputMode::TEXT_MONITOR)
          Serial.println("# Stop or finish capture before binary mode");
      } else if (packetQueue == nullptr) {
        if (outputMode == OutputMode::TEXT_MONITOR)
          Serial.println("# Binary mode unavailable: packet queue allocation failed");
      } else {
        Serial.println("# Switching to binary packet version 3");
        Serial.flush();
        clearPacketQueue();
        outputMode = OutputMode::BINARY_FULL_RATE;
      }
      break;

    case 'f': case 'F': setOperatingMode(OperatingMode::FULL_COLLECTION); break;
    case 'k': case 'K': setOperatingMode(OperatingMode::BUTTON_TEST); break;
    case 'a': case 'A': setOperatingMode(OperatingMode::IMU_TEST); break;
    case 'l': case 'L': setOperatingMode(OperatingMode::LRA_TEST); break;
    case 's': case 'S': setOperatingMode(OperatingMode::SYNC_TEST); break;

    case '\r': case '\n': case ' ': case '\t': break;
    default:
      if (outputMode == OutputMode::TEXT_MONITOR)
        Serial.printf("# Unknown command '%c'; send h for help\n", command);
      break;
  }
}

void updateRecordButton(uint32_t nowMs) {
  if (recordButton.wasPressed()) {
    recordButtonPressedAtMs = nowMs;
    recordLongPressHandled = false;
  }

  // In component-test modes the record button is observed only; it must not
  // accidentally start calibration or a recording.
  if (operatingMode != OperatingMode::FULL_COLLECTION &&
      operatingMode != OperatingMode::SYNC_TEST) {
    return;
  }

  if (recordButton.isPressed() && !recordLongPressHandled &&
      static_cast<uint32_t>(nowMs - recordButtonPressedAtMs) >=
          RECORD_LONG_PRESS_MS &&
      systemState != SystemState::RECORDING) {
    startCalibration();
    recordLongPressHandled = true;
  }

  if (recordButton.wasReleased() && !recordLongPressHandled &&
      systemState != SystemState::CALIBRATING) {
    toggleRecording();
  }
}

void updateInputs(uint32_t nowMs) {
  morseButton.update(nowMs);
  recordButton.update(nowMs);
  updateRecordButton(nowMs);

  while (Serial.available() > 0) {
    processSerialCommand(static_cast<char>(Serial.read()));
  }
}

uint8_t makeStatusFlags() {
  uint8_t flags = 0;
  if (imuReady) flags |= 0x01;
  if (timingOverrunCount > 0) flags |= 0x02;
  if (calibrationIsValid()) flags |= 0x04;
  if (drvReady) flags |= 0x08;
  if (i2cErrorCount > 0) flags |= 0x10;
  if (packetQueueOverflowCount > 0) flags |= 0x20;
  return flags;
}

void sendTextSample(const SamplePacket& packet) {
  const uint32_t decimation = IMU_OUTPUT_RATE_HZ / TEXT_OUTPUT_RATE_HZ;
  if (decimation > 1 && (packet.sampleIndex % decimation) != 0) return;

  Serial.printf("%lu,%lu,%lu,%llu,%d,%d,%d,%d,%d,%d,%u,%u,%u,%u\n",
                static_cast<unsigned long>(packet.recordingId),
                static_cast<unsigned long>(packet.calibrationId),
                static_cast<unsigned long>(packet.sampleIndex),
                static_cast<unsigned long long>(packet.timestampUs),
                packet.accelX, packet.accelY, packet.accelZ,
                packet.gyroX, packet.gyroY, packet.gyroZ,
                packet.keyState, packet.motorCommand,
                packet.systemState, packet.statusFlags);
}

void processCapturedSample(const ImuRawSample& imu, uint64_t acquiredUs) {
  if (previousSampleUs != 0) {
    const uint64_t intervalUs = acquiredUs - previousSampleUs;
    if (intervalUs > static_cast<uint64_t>(NOMINAL_SAMPLE_PERIOD_US) * 3ULL / 2ULL) {
      const uint32_t periods = static_cast<uint32_t>(
          (intervalUs + NOMINAL_SAMPLE_PERIOD_US / 2) / NOMINAL_SAMPLE_PERIOD_US);
      if (periods > 1) timingOverrunCount += periods - 1;
    }
  }
  previousSampleUs = acquiredUs;

  if (systemState == SystemState::CALIBRATING && outputMode == OutputMode::TEXT_MONITOR) {
    addCalibrationSample(imu);
  }

  SamplePacket packet{};
  packet.magic = PACKET_MAGIC;
  packet.version = PACKET_VERSION;
  packet.packetType = PACKET_TYPE_SAMPLE;
  packet.recordingId = systemState == SystemState::RECORDING ? recordingId : 0;
  packet.calibrationId = calibrationId;
  packet.sampleIndex = sampleIndex;
  packet.timestampUs = acquiredUs - captureStartUs;
  packet.accelX = imu.accelX;
  packet.accelY = imu.accelY;
  packet.accelZ = imu.accelZ;
  packet.gyroX = imu.gyroX;
  packet.gyroY = imu.gyroY;
  packet.gyroZ = imu.gyroZ;
  packet.keyState = effectiveKeyPressed() ? 1 : 0;
  packet.motorCommand = currentMotorCommand;
  packet.systemState = static_cast<uint8_t>(systemState);
  packet.statusFlags = makeStatusFlags();
  packet.crc16 = crc16CcittFalse(reinterpret_cast<const uint8_t*>(&packet),
                                sizeof(packet) - sizeof(packet.crc16));

  if (outputMode == OutputMode::TEXT_MONITOR) sendTextSample(packet);
  else enqueuePacket(packet);

  ++sampleIndex;
  ++packetsProduced;
}

void serviceAcquisition(uint64_t nowUs) {
  if (!isCapturing() || !imuReady) return;

  if (systemState == SystemState::CALIBRATING && nowUs >= calibrationEndUs) {
    finishCalibration();
    return;
  }

  if (!imuDataReady()) {
    ++imuNotReadyCount;
    return;
  }

  ImuRawSample sample{};
  if (!readImuRaw(sample)) return;
  processCapturedSample(sample, nowMicroseconds());
}

void serviceButtonTest() {
  if (morseButton.isPressed() != previousTestMorseState) {
    previousTestMorseState = morseButton.isPressed();
    Serial.printf("# BUTTON,morse=%u\n", previousTestMorseState ? 1 : 0);
  }
  if (recordButton.isPressed() != previousTestRecordState) {
    previousTestRecordState = recordButton.isPressed();
    Serial.printf("# BUTTON,record=%u\n", previousTestRecordState ? 1 : 0);
  }
}

void serviceImuTest(uint64_t nowUs) {
  if (!imuReady || !imuDataReady()) return;
  ImuRawSample sample{};
  if (!readImuRaw(sample)) return;

  const uint64_t periodUs = 1000000ULL / IMU_TEST_OUTPUT_RATE_HZ;
  if (nowUs - lastImuTestPrintUs < periodUs) return;
  lastImuTestPrintUs = nowUs;
  Serial.printf("# IMU,ax=%d,ay=%d,az=%d,gx=%d,gy=%d,gz=%d\n",
                sample.accelX, sample.accelY, sample.accelZ,
                sample.gyroX, sample.gyroY, sample.gyroZ);
}

void serviceTestModes(uint64_t nowUs) {
  if (outputMode != OutputMode::TEXT_MONITOR) return;
  switch (operatingMode) {
    case OperatingMode::BUTTON_TEST: serviceButtonTest(); break;
    case OperatingMode::IMU_TEST: serviceImuTest(nowUs); break;
    case OperatingMode::LRA_TEST: updateMotorFromKey(); break;
    default: break;
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(800);

  morseButton.begin();
  recordButton.begin();

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, I2C_CLOCK_HZ);
  Wire.setTimeOut(20);

  Serial.println();
  Serial.println("# Morse-vibration hardware firmware starting");
  Serial.printf("# pins,morse=%u,record=%u,sda=%u,scl=%u\n",
                PIN_MORSE_BUTTON, PIN_RECORD_BUTTON,
                PIN_I2C_SDA, PIN_I2C_SCL);

  imuReady = beginImu();
  drvReady = beginDriver();
  allocatePacketQueue();

  if (!imuReady || !drvReady) {
    Serial.println("# Hardware startup incomplete. Use k/a/l test modes after checking wiring.");
  }

  printHelp();
  printStatus();
}

void loop() {
  const uint64_t nowUs = nowMicroseconds();
  const uint32_t nowMs = static_cast<uint32_t>(nowUs / 1000ULL);

  updateInputs(nowMs);
  updateMotorFromKey();
  serviceAcquisition(nowUs);
  drainPacketQueue();
  serviceTestModes(nowUs);

  delayMicroseconds(20);
}
