
from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(df, contamination=0.1):
    # Add category and month features if available
    features = ['Amount']
    if 'Category' in df.columns:
        cat_dummies = pd.get_dummies(df['Category'], prefix='Cat')
        df = pd.concat([df, cat_dummies], axis=1)
        features += [col for col in df.columns if col.startswith('Cat_')]
    if 'Date' in df.columns:
        df['Month'] = df['Date'].dt.month
        features.append('Month')
    df_model = df[features]
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(df_model)
    # Add severity score (decision function)
    scores = iso.decision_function(df_model)
    df['Anomaly'] = ['Anomaly' if p == -1 else 'Normal' for p in iso_pred]
    df['AnomalyScore'] = scores
    return df