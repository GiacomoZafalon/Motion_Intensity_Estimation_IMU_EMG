#include <WiFi.h>

const char* ssid = "YourWifiSSID";
const char* password = "YourWifiPassword";

WiFiClient client;
const char* host = "192.168.43.19"; // IP address of your computer running a server to receive sensor data
const int port = 3335; // Port number on which your server is listening (3335)

const int pinEMG1 = A0;
const int pinEMG2 = A1;
const int pinEMG3 = A2;
int signalEMG = 0;

void setup() {
  Serial.begin(115200);

  // Connect to Wi-Fi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
}

void loop() {
  signalEMG1 = analogRead(pinEMG1);
  signalEMG2 = analogRead(pinEMG2);
  signalEMG3 = analogRead(pinEMG3);

  // Connect to the server
  if (!client.connected()) {
    Serial.print("Connecting to ");
    Serial.println(host);
    if (client.connect(host, port)) {
      Serial.println("Successfully connected :)");
    } else {
      Serial.println("Connection stopped :( Start the server");
      // Close the connection
      client.stop();
      while(!client.connected()) {
        if (client.connect(host, port)){
          Serial.println("Successfully connected :)");
          break;
        }
      }
    }
  }

  String dataString = String(signalEMG1) + ", " + String(signalEMG2) + ", " + String(signalEMG3);

  Serial.println(dataString);
  client.println(dataString);
  delay(10);

}
