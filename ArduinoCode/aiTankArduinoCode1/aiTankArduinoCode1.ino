#include "DeviceDriverSet_xxx0.h"
#include "ApplicationFunctionSet_xxx0.cpp"

DeviceDriverSet_Motor AppMotor;
int baseSpeed = 150;
int leftSpeed = baseSpeed;
int rightSpeed = 150;  // Calibrated for balance

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
      AppMotor.DeviceDriverSet_Motor_control(direction_just, leftSpeed,
                                             direction_just, rightSpeed,
                                             control_enable);
    } else if (command == "STOP") {
      AppMotor.DeviceDriverSet_Motor_control(direction_void, 0,
                                             direction_void, 0,
                                             control_enable);
    } else if (command == "LEFT") {
      // LEFT = left forward slow, right backward fast
      AppMotor.DeviceDriverSet_Motor_control(direction_just, leftSpeed - 60,
                                             direction_back, rightSpeed - 50,
                                             control_enable);
    } else if (command == "RIGHT") {
      // RIGHT = left backward fast, right forward slow
      AppMotor.DeviceDriverSet_Motor_control(direction_back, leftSpeed - 50,
                                             direction_just, rightSpeed - 60,
                                             control_enable);
    }

    else if (command == "BACKWARD") {
      AppMotor.DeviceDriverSet_Motor_control(direction_back, baseSpeed,
                                             direction_back, baseSpeed,
                                             control_enable);
    }
    
  }
}