import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def remove_outliers(series, z_thresh=3):
    mean=series.mean()
    std=series.std()
    return series[(np.abs(series - mean) < z_thresh * std)]

def predict_future_expense(df, lookback_days=60):
    df=df.sort_values("Date")
    df=df.copy()
    df['Day']=np.arange(len(df))
    df['Month']=df['Date'].dt.month
    if 'Category' in df.columns:
        cat_dummies=pd.get_dummies(df['Category'], prefix='Cat')
        df=pd.concat([df, cat_dummies], axis=1)
    df['Amount_lag1']=df['Amount'].shift(1).fillna(0)
    df['Amount_lag2']=df['Amount'].shift(2).fillna(0)
    if len(df) > lookback_days:
        df=df.iloc[-lookback_days:]
        df['Day']=np.arange(len(df))
    y_no_out=remove_outliers(df['Amount'])
    feature_cols=['Day', 'Month', 'Amount_lag1', 'Amount_lag2']
    cat_cols=[col for col in df.columns if col.startswith('Cat_')]
    feature_cols += cat_cols
    X=df.loc[y_no_out.index, feature_cols]
    y=y_no_out
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Save the exact feature columns used for fitting
    fit_feature_cols=list(X.columns)
    last_row=df.iloc[-1]
    future_X=[]
    for i in range(1, 6):
        row={
            'Day': last_row['Day'] + i,
            'Month': ((last_row['Date'] + pd.Timedelta(days=i)).month if 'Date' in last_row else 1),
            'Amount_lag1': last_row['Amount'],
            'Amount_lag2': last_row['Amount_lag1']
        }
        for col in cat_cols:
            row[col]=last_row[col] if col in last_row else 0
        future_X.append(row)
    future_X_df=pd.DataFrame(future_X)
    # Add missing columns and enforce order
    for col in fit_feature_cols:
        if col not in future_X_df.columns:
            future_X_df[col]=0
    # Remove any extra columns
    future_X_df=future_X_df[fit_feature_cols]
    # Ensure all values are scalars and convert to float
    future_X_df=future_X_df.map(lambda x: x if np.isscalar(x) else 0)
    future_X_df=future_X_df.astype(float)
    predictions=model.predict(future_X_df)
    return predictions