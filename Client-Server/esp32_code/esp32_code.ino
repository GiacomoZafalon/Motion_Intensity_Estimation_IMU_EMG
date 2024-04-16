#include <WiFi.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <Wire.h>

#define BNO055_DELAY_MS (10)

const char* ssid = "Jack";
const char* password = "Ck710tete";

WiFiClient client;
const char* host = "192.168.43.19"; // IP address of your computer running a server to receive sensor data
const int port = 3334; // Port number on which your server is listening (Sensor1-3331, Sensor2-3332, Sensor3-3333, Sensor4-3334)

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29);

void setup() {
  Serial.begin(115200);
  delay(10);
  
  // Connect to Wi-Fi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  bno.begin();
}

void loop() {
  // Read sensor data
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  imu::Quaternion quat = bno.getQuat();
  sensors_event_t acc, gyr, mag, linAcc;

  bno.getEvent(&acc, Adafruit_BNO055::VECTOR_ACCELEROMETER);
  bno.getEvent(&gyr, Adafruit_BNO055::VECTOR_GYROSCOPE);
  bno.getEvent(&mag, Adafruit_BNO055::VECTOR_MAGNETOMETER);
  bno.getEvent(&linAcc, Adafruit_BNO055::VECTOR_LINEARACCEL);

  // Send sensor data over network
  sendSensorData(euler.x(), euler.y(), euler.z(), acc.acceleration.x, acc.acceleration.y, acc.acceleration.z, gyr.gyro.x, gyr.gyro.y, gyr.gyro.z, mag.magnetic.x, mag.magnetic.y, mag.magnetic.z, linAcc.acceleration.x, linAcc.acceleration.y, linAcc.acceleration.z, quat.w(), quat.x(), quat.y(), quat.z());

  delay(BNO055_DELAY_MS); // Adjust delay as needed
}

void sendSensorData(float euler_x, float euler_y, float euler_z, float accel_x, float accel_y, float accel_z, float gyro_x, float gyro_y, float gyro_z, float mag_x, float mag_y, float mag_z, float linear_accel_x, float linear_accel_y, float linear_accel_z, float quat_w, float quat_x, float quat_y, float quat_z) {
  static double timestep;

  // Connect to the server
  if (!client.connected()) {
    Serial.print("Connecting to ");
    Serial.println(host);
    if (client.connect(host, port)) {
      Serial.println("Successfully connected :)");
      // Set the timestep to 0
      timestep = 0;
    } else {
      Serial.println("Connection stopped :( Start the server");
      // Close the connection
      client.stop();
      while(!client.connected()) {
        if (client.connect(host, port)){
          Serial.println("Successfully connected :)");
          // Reset the timestep to 0
          timestep = 0;
          break;
        }
      }
    }
  }

  // Increase the timestep
  timestep += BNO055_DELAY_MS/1000.0;

  // Send sensor data to the server
  String dataString = String(timestep) + ", " + String(euler_x) + ", " + String(euler_y) + ", " + String(euler_z) + ", " +
                      String(accel_x) + ", " + String(accel_y) + ", " + String(accel_z) + ", " +
                      String(gyro_x) + ", " + String(gyro_y) + ", " + String(gyro_z) + ", " +
                      String(mag_x) + ", " + String(mag_y) + ", " + String(mag_z) + ", " +
                      String(linear_accel_x) + ", " + String(linear_accel_y) + ", " + String(linear_accel_z) + ", " +
                      String(quat_w) + ", " + String(quat_x) + ", " + String(quat_y) + ", " + String(quat_z);
  client.println(dataString);
}
