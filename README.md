# 12-lead ECG classification

## Model

The model ```final_model.hdf5``` used in this work is a convolutional neural network fine-tuned on a dataset of patients with a cardiac arrhythmia.
If you want to use the model, which is in the hdf5 format, you can simply load it in your code using ```keras.models``` with the ```load_model``` function.

## Dataset

The model was trained on a 12-lead ECG dataset provided by the IKEM hospital. Dataset is not publicly available. If you are interested, please contact us.

## Installation

Make sure that your virtual environment satisfies the following requirements before running any code:

* Python version: `>=3.10`
* Dependencies: `pip install -r requirements.txt`