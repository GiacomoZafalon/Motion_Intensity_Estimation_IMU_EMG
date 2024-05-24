const int pinEMG1 = A0;  // Analog input pin for EMG signal
// const int pinEMG2 = A1;  // Analog input pin for EMG signal
// const int pinEMG3 = A2;  // Analog input pin for EMG signal

void setup() {
  Serial.begin(115200);
}

void loop() {
  int EMGValue1 = analogRead(pinEMG1);
  // int EMGValue2 = analogRead(pinEMG2);
  // int EMGValue3 = analogRead(pinEMG3);
  Serial.println(EMGValue1);
  // Serial.print(",");
  // Serial.print(EMGValue2);
  // Serial.print(",");
  // Serial.println(EMGValue3);
  delay(100);  // Adjust delay as needed
}