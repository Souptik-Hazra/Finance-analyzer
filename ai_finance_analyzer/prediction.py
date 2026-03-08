import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def remove_outliers(series, z_thresh=3):
    mean = series.mean()
    std = series.std()
    return series[(np.abs(series - mean) < z_thresh * std)]

def predict_future_expense(df, lookback_days=60):
    df = df.sort_values("Date")
    df = df.copy()
    df['Day'] = np.arange(len(df))
    # Use only recent data
    if len(df) > lookback_days:
        df = df.iloc[-lookback_days:]
        df['Day'] = np.arange(len(df))
    # Remove outliers
    y_no_out = remove_outliers(df['Amount'])
    X = df.loc[y_no_out.index, ['Day']]
    y = y_no_out
    # Use RandomForest for robust regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    future_days = np.array([[df['Day'].max() + i] for i in range(1, 6)])
    predictions = model.predict(future_days)
    return predictions