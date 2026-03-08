from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    model = IsolationForest(contamination=0.1)
    df_model = df[['Amount']]
    model.fit(df_model)
    df['Anomaly'] = model.predict(df_model)
    df['Anomaly'] = df['Anomaly'].map({1:"Normal",-1:"Anomaly"})
    return df