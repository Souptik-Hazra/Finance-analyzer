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
    prev_amount=last_row['Amount']
    prev_amount_lag1=last_row['Amount_lag1'] if 'Amount_lag1' in last_row else 0
    for i in range(1, 6):
        row={
            'Day': last_row['Day'] + i,
            'Month': ((last_row['Date'] + pd.Timedelta(days=i)).month if 'Date' in last_row else 1),
            'Amount_lag1': prev_amount,
            'Amount_lag2': prev_amount_lag1
        }
        for col in cat_cols:
            row[col]=last_row[col] if col in last_row else 0
        future_X.append(row)
        # Predict for this row
        future_X_df=pd.DataFrame([row])
        for col in fit_feature_cols:
            if col not in future_X_df.columns:
                future_X_df[col]=0
        future_X_df=future_X_df[fit_feature_cols]
        future_X_df=future_X_df.map(lambda x: x if np.isscalar(x) else 0)
        future_X_df=future_X_df.astype(float)
        pred=model.predict(future_X_df)[0]
        prev_amount_lag1=prev_amount
        prev_amount=pred
    # Collect all predictions
    # Instead of re-predicting, store predictions as we generate them
    predictions=[]
    prev_amount=last_row['Amount']
    prev_amount_lag1=last_row['Amount_lag1'] if 'Amount_lag1' in last_row else 0
    # Get most frequent category from history
    if 'Category' in df.columns:
        most_freq_cat=df['Category'].mode()[0]
        cat_dummy_cols=[col for col in fit_feature_cols if col.startswith('Cat_')]
    else:
        most_freq_cat=None
        cat_dummy_cols=[]
    # Simulate realistic future days
    for i in range(1, 6):
        # Increment month if day exceeds 30
        future_day=last_row['Day'] + i
        future_month=((last_row['Date'] + pd.Timedelta(days=i)).month if 'Date' in last_row else 1)
        row={
            'Day': future_day,
            'Month': future_month,
            'Amount_lag1': prev_amount,
            'Amount_lag2': prev_amount_lag1
        }
        # Simulate category: cycle through most frequent or random
        for col in cat_dummy_cols:
            row[col]=0
        if most_freq_cat:
            cat_col='Cat_' + str(most_freq_cat)
            if cat_col in cat_dummy_cols:
                row[cat_col]=1
        future_X_df=pd.DataFrame([row])
        for col in fit_feature_cols:
            if col not in future_X_df.columns:
                future_X_df[col]=0
        future_X_df=future_X_df[fit_feature_cols]
        future_X_df=future_X_df.map(lambda x: x if np.isscalar(x) else 0)
        future_X_df=future_X_df.astype(float)
        pred=model.predict(future_X_df)[0]
        predictions.append(pred)
        prev_amount_lag1=prev_amount
        prev_amount=pred
    return predictions