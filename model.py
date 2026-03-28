import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def get_data(stock):
    df = yf.download(stock, start="2020-01-01", end="2024-01-01")
    return df

def prepare_data(df):
    df = df[['Close']]
    df['Prediction'] = df['Close'].shift(-30)

    X = np.array(df[['Close']])[:-30]
    y = np.array(df['Prediction'])[:-30]

    return X, y

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

def predict_future(model, df):
    future = df[['Close']].tail(30)
    predictions = model.predict(future)
    return predictions