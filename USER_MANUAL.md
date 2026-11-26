# User Manual

## Application
6-Axis Industrial Robotic Arm Tutor & Simulator

## Requirements
- Windows 11
- Python 3.11 recommended
- Internet connection for package installation; optional for live AI API use

## Installation
Follow the steps in README.md.

## How to use
1. Start the app with `py -m streamlit run app.py`.
2. Open the local address shown in the terminal (usually http://localhost:8501).
3. In **3D Simulator**, move J1–J6 sliders.
4. Observe the 3D arm and live tool XYZ position.
5. In **Inverse Kinematics**, enter a target XYZ position and calculate joint angles.
6. In **AI Tutor**, ask a robotics question. Built-in explanations work without an API key. Live AI requires your own OpenAI API key.
7. Use **Project Guide** to review the method and limitations.

## Suggested screenshots for submission
Capture:
- Full 3D simulator with six joint controls
- Live XYZ position
- Inverse kinematics result and target marker
- AI tutor answer
- Project Guide / limitations page

## Troubleshooting
- If `py` is not recognized, reinstall Python and select Add Python to PATH.
- If Streamlit is not recognized, use `py -m streamlit run app.py`.
- If packages fail to install, run `py -m pip install --upgrade pip` then install requirements again.
