# 🏡 House Price Prediction using Linear Regression

## Table of Contents
- [Project Overview](#project-overview)  
- [Features](#features)  
- [How It Works](#how-it-works)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dataset](#dataset)  
- [Model Evaluation](#model-evaluation)  
- [Future Improvements](#future-improvements)  
- [Author](#author)  

---

## Project Overview

This project is a **House Price Prediction system** implemented in Python using **Linear Regression**.  
The system predicts house prices based on **features like size (sqft), number of rooms, and location**.  

It demonstrates basic **machine learning workflows**:
- Data preprocessing  
- One-hot encoding categorical features  
- Train-test split  
- Model training and evaluation  
- Making predictions on new data  

---

## Features

- Predict house prices based on **size, rooms, and location**  
- Handles **categorical data** using one-hot encoding  
- Calculates **Root Mean Squared Error (RMSE)** for model evaluation  
- Visualizes **actual vs predicted prices** using Matplotlib  
- Easy to customize for **larger datasets**  

---

## How It Works

1. **Data Preparation**  
   - Sample dataset is created or a CSV can be loaded.  
   - Features (`Size_sqft`, `Rooms`, `Location`) and target (`Price`) are separated.  

2. **Encoding Categorical Features**  
   - `Location` is one-hot encoded using `ColumnTransformer` and `OneHotEncoder`.  

3. **Model Training**  
   - Data is split into **training** and **testing** sets.  
   - A **Linear Regression** model is trained on the training set.  

4. **Prediction and Evaluation**  
   - Model predicts house prices on the test set.  
   - RMSE is calculated to measure prediction accuracy.  
   - Scatter plot of **actual vs predicted prices** is displayed.  

5. **Example Prediction**  
   - Predicts price for a new input: `1500 sqft, 3 rooms, Location B`.  

---

## Installation

1. Clone the repository or download the script.  

2. Install required Python packages:

```bash
pip install pandas scikit-learn matplotlib numpy
