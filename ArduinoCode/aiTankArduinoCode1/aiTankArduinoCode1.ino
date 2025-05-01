#include "DeviceDriverSet_xxx0.h"
#include "ApplicationFunctionSet_xxx0.cpp"

DeviceDriverSet_Motor AppMotor;
int speed = 150;

void setup() {
  Serial.begin(9600);
  AppMotor.DeviceDriverSet_Motor_Init();
  Serial.println("Arduino: successfully connected");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    Serial.print("Received: ");
    Serial.println(command);

    if (command == "FORWARD") {
      AppMotor.DeviceDriverSet_Motor_control(direction_just, speed,
                                             direction_just, speed,
                                             control_enable);
    } else if (command == "STOP") {
      AppMotor.DeviceDriverSet_Motor_control(direction_void, 0,
                                             direction_void, 0,
                                             control_enable);
    } else if (command == "LEFT") {
      // Turn in place left: left motor backward, right motor forward
      AppMotor.DeviceDriverSet_Motor_control(direction_back, speed,
                                             direction_just, speed,
                                             control_enable);
    } else if (command == "RIGHT") {
      // Turn in place right: left motor forward, right motor backward
      AppMotor.DeviceDriverSet_Motor_control(direction_just, speed,
                                             direction_back, speed,
                                             control_enable);
    }
  }
}
