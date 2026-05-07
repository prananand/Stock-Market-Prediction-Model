## Stock Prediction Model

This project is a Streamlit app that predicts the next trading day stock price for a ticker symbol entered by the user.

## How it works
. Downloads stock data from yfinance
. Creates feature engineering like moving averages, RSI, MACD, and Bollinger Bands
. Trains machine learning models
. Compares Linear Regression, Random Forest, Gradient Boosting, and an Average Model
. Predicts whether the stock may go up or down
. Saves model results into a SQLite database
. Displays charts showing prediction errors and model results

## How To Run
pip install -r ReadMe\&Requirments/requirements.txt
streamlit run App/app.py
