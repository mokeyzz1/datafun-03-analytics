# datafun-03-analytics

This project demonstrates a professional approach to analytics using Git, Python, a virtual environment, and data processing libraries. It includes scripts to load, preprocess, visualize data, and train a machine learning model.

## Project Overview

The `moses_main.py` script performs the following tasks:
1. **Data Loading**: Reads data from a CSV file.
2. **Data Preprocessing**: Cleans the data by removing missing values.
3. **Data Visualization**: Creates histograms to visualize feature distributions.
4. **Model Training**: Trains a RandomForest model and evaluates its accuracy.

## Create and Activate Project Virtual Environment

### For Mac
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Freeze Requirements

```shell
py -m pip freeze > requirements.txt

## Git Add / Commit / Push 

```shell
git add .
git commit -m "add .gitignore, commands to README.md"
git push -u origin main
```