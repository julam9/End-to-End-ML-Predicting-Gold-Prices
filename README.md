### 🏷️ Gold Price Forecasting with End to End Machine Learning

## 📋 Table of Contents 
- [📈 Introduction](#-introduction)
- [🗂️ Project Structure](#️-project-structure)
- [⭐ Features](#-features)
- [⚙️ Installation](#-installation)
- [🚀 How to Use](#-how-to-use)
- [📊 Streamlit Dashboard](#-streamlit-dashboard)
- [🔄 Project Workflow](#-project-workflow)
- [🔑 API & Data Source](#-api--data-source)
- [📉 Model & Evaluation](#-model--evaluation)
- [🚧 Future Improvements](#-future-improvements)
- [📄 License](#-license)

## 📖 Introduction
This project is my first end to end machine learning project. Through this project, I learn how end to end works along with machine learning. I choose the topic of forecasting gold price since gold become hot topic and the price is rising quite significantly. 

## 🗂️ Project Structure
- scr/ : A folder contains module used for end to end process, like data collection, data preprocessing, data validation, model building, model saving, model loading, model evaluation
- models/ : A folder contains saved model, the model named with timestamp of model constructed. This folder will be used for pipeline validation.
- notebooks/ : A folder contains notebook that used to test the code and do some exploration
- data/ : A folder contains data in CSV format for visualization and training. These data are used as anticipation of Yahoo Finance API Error (Rate Limit or other error)
- .env : Contains API Key from Alpha Vantage to collect data
- .gitignore : Contains files that won't be pushed to github
- poetry.lock : Contains store project metadata, build system information, and dependency specifications 
- pyproject.toml :  Contains package version needed to be installed 
- pipeline_validation.py : Script to check if the entire end to end process is working
- backup_data.py : Script to fetch and save data to anticipate if the yahoo finance API got error
- requirements.txt : Contains package version needed (only for streamlit deployment)
- dashboard.py : Script to deploy the dashboard about price visualization and forecasting 


## 🚀 Features
- Load gold data using Yahoo Finance API (using local data if needed)
- Feature engineering and k-best feature selection 
- Forecasting gold price using XGBoost model 
- Interactive forecasting dashboard (choose date range, forecast length, feature count) 

## ⚙️ Installation
To use this repo you can do these :
```bash 
git clone https://github.com/julam9/End-to-End-ML-Predicting-Gold-Prices 
cd predict-gold-prices
poetry install
```
## ▶️ How to Use
You can try the pipeline validation script :
```bash 
poetry run python pipeline_validation.py
```
## 🖥️ Streamlit Dashboard
You can launch the dashbnoard by doing this :
```bash
poetry run streamlit run dashboard.py
```
## 🔁 Project Workflow
- Load data from API
- Clean and validate data
- Add features
- Train model
- Evaluate performance
- Visualize results in Streamlit

## 🗃️ API / Data Source
This project is using Yahoo Finance API. You can use it by installing yfinance package.

## 📊 Model & Evaluation
- Model: XGBoost Regressor
- Evaluation: MAE and RMSE

## 💡 Future Improvements
- Other feature engineering for forecasting price 
- Try another forecasting model, ie prophet
- Improve dashboard by including other elements 

## 📄 License
This project is licensed under the MIT License.