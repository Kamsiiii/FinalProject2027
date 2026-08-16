Hand Gesture Detection with the Leap Motion Controller
Kamsi Chibueze 

Requirements
•	Ultraleap Leap Motion Controller (USB) with Control Panel v4.1.0
•	Anaconda or Miniconda (Python 3)
•	Windows 10/11

Note: This project uses the v4 Control Panel's native WebSocket API

Setup
1.	Enable WebSocket in the Control Panel. Open the Leap Motion Control Panel, then go to General tab, then tick "Allow Web Apps". Click Apply.
2.	Install dependencies (Anaconda Prompt):

conda activate base
pip install numpy scikit-learn tensorflow joblib matplotlib pandas websocket-client pillow

Running the Project
Always use Anaconda Prompt, not PowerShell or CMD.
If models are already trained:

conda activate base
cd C:\FinalProject2027
python app.py

Click Start, then sign ASL letters in front of the sensor.

If starting from scratch, run in this order:
1.	python collect_data.py - Collect static letters A–Y (excluding J and Z) 
2.	python collect_dynamic.py - Collect J and Z sequences
3.	python collect_neither.py - Collect static hold sequences
4.	python train_model.py - Train the Random Forest classifier
5.	python train_dynamic.py - Train the LSTM for J and Z
6.	python app.py - Launch the GUI application

GUI Controls
	Start / Stop: Toggle recognition on/off
	Space: Commit current word
	Backspace: Remove last letter
	Confidence slider: Adjust recognition threshold
	Export: Save session log to file
