# Technical Report (editable starter draft)

## Title
AI-Powered 6-Axis Industrial Robotic Arm Tutor & Simulator

## Abstract
This project presents an educational Python application for visualizing a generic six-revolute-joint robotic arm. The application provides joint-angle controls, three-dimensional visualization, forward kinematics, numerical position-only inverse kinematics, and a robotics learning assistant. It is intended to demonstrate foundational industrial robotics concepts in an accessible desktop-browser interface.

## 1. Introduction
Industrial articulated robots commonly use multiple rotary joints to position and orient an end effector. Understanding the relationship between joint variables and tool pose is central to robot programming and automation. This project provides an interactive simulator for learning these concepts.

## 2. Objectives
- Simulate a generic six-axis articulated robot.
- Calculate tool position from joint angles using forward kinematics.
- Estimate joint angles for a requested XYZ tool position using numerical inverse kinematics.
- Visualize the robot in 3D.
- Provide explanations of basic robotics concepts.

## 3. Methodology
The robot is represented as a serial chain of six revolute joints. The application uses an illustrative standard Denavit–Hartenberg parameter table. Homogeneous transformation matrices are multiplied to calculate the tool pose. SciPy least-squares optimization is used to minimize XYZ position error for inverse kinematics. Plotly displays the links and tool coordinate axes. Streamlit provides the user interface.

## 4. Software and tools
Python, Streamlit, NumPy, SciPy, Plotly, and optionally the OpenAI API.

## 5. Results
**Complete this section only after running the software.** Insert screenshots of the simulator, joint controls, tool coordinates, inverse-kinematics output, and AI tutor. Record test inputs and observed position errors.

## 6. Limitations
The robot geometry is illustrative and is not a manufacturer-specific model. The inverse-kinematics demonstration targets XYZ position only and does not constrain tool orientation. Collision checking, robot dynamics, payload calculations, calibration, and physical controller communication are outside the current prototype. The built-in tutor provides fixed educational explanations unless an API key is configured.

## 7. Future scope
Future work may add full-pose inverse kinematics, collision detection, trajectory generation, robot-specific URDF models, ROS 2/MoveIt integration, camera-based perception, gripper simulation, and carefully validated hardware integration.

## 8. Conclusion
The prototype demonstrates how Python can combine robot kinematics, visualization, and AI-assisted learning in one interactive educational application. It provides a foundation for further robotics study and development.
