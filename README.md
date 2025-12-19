Physics-Informed Neural Network for Heart Rate Recovery

This repository contains a PyTorch implementation of a Physics-Informed
Neural Network (PINN) used to model heart rate recovery after exercise.

The model is trained on experimental heart rate data and constrained
by a nonlinear ordinary differential equation.


System Requirements
-------------------
- Python 3.9 or newer
- Windows, macOS, or Linux
- CPU is sufficient (GPU not required)


Required Python Packages
------------------------
- torch
- pandas
- matplotlib
- openpyxl
- numpy


Files
-----
- pinn_submission.py   : Main Python script
- data/trial1_fall.xlsx: Heart rate data file
- README.md            : Instructions (this file)


How to Run
----------
1. Place the data file in the following path:
   data/trial1_fall.xlsx

2. Install dependencies:
   pip install torch pandas matplotlib openpyxl numpy


3. Run the script:
   python pinn_submission.py


Data Format
-----------
The Excel file must contain the following columns:
- delta_t_s : Time after exercise (seconds)
- bpm       : Heart rate (beats per minute)


Output
------
- Training loss is printed during execution
- A plot is displayed showing heart rate vs time
  comparing data and the PINN model prediction


Author
------
Aaron Yuan
