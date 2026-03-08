import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def predict_future_expense(df):
    df = df.sort_values("Date")
    df['Day'] = np.arange(len(df))
    X = df[['Day']]
    y = df['Amount']
    model = LinearRegression()
    model.fit(X,y)
    future_days = np.array([[len(df)+i] for i in range(1,6)])
    predictions = model.predict(future_days)
    return predictions