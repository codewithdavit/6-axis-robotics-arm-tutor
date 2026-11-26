# 6-Axis Industrial Robotic Arm Tutor & Simulator

A Windows-friendly educational prototype built with Python, Streamlit, Plotly, NumPy and SciPy.

## Run on Windows 11

1. Install Python 3.11 from https://www.python.org/downloads/
   - During installation, enable **Add Python to PATH**.
2. Extract this ZIP folder.
3. Open the extracted folder in VS Code or File Explorer.
4. Open PowerShell in this folder.
5. Create a virtual environment (recommended):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```cmd
   .venv\Scripts\activate.bat
   ```

6. Install packages:
   ```powershell
   py -m pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   py -m streamlit run app.py
   ```
8. A browser tab will open with the simulator.

## Features
- Six independent revolute-joint sliders
- Interactive 3D robot visualization
- Forward kinematics using standard DH transformations
- Numerical inverse kinematics for XYZ tool position
- Built-in robotics explanations
- Optional OpenAI API-powered tutor (requires your own API key and may incur API charges)

## Demonstration
- Move the six joint sliders.
- Observe the robot pose and live tool XYZ position.
- Enter a target XYZ position in the Inverse Kinematics tab.
- Calculate a joint solution and apply it to the simulator.
- Ask the tutor to explain forward kinematics, inverse kinematics, six axes or DH parameters.

## Important limitations and safety
This is an educational simulator using illustrative generic robot geometry. It is not a manufacturer's calibrated robot model, certified industrial controller, or safety-rated system. It does not implement collision checking, dynamics, payload limits, safety-rated monitoring, or physical robot communication. Do not use it to control real machinery.
