# Code and data for "Rational use of cognitive resources in human planning"

## Data
Data is available in pickle format, which can be loaded in Python. For each experiment, trials.pkl contains one row for each trial (excluding practice), participants.pkl contains one row for each participant, and trials.json contains the trials in a simplified json format for use by the model.

## Model
model/steps.sh specifies the sequence of steps necessary to recreate model results. However, note that the modeling is very computationally intensive and will take days or weeks to run on a laptop or typical desktop. If you are interested in using this code, I encourage you to contact me beforehand. It is not well documented at present.

## Analysis
This code produces all the figures and statistics reported in the paper. Each experiment has a main file, e.g. analysis/exp1.py. The contents of this file must be pasted into an IPython interpreter.

NOTE: to run the analysis code, you will need the model results files, which are too large to be stored github. They can be found [on osf](https://osf.io/6venh/).
