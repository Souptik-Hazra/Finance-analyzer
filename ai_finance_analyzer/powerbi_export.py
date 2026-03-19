import requests
import pandas as pd

def export_to_powerbi(df, powerbi_url):
    """
    Send a DataFrame to Power BI streaming dataset API.
    Args:
        df (pd.DataFrame): DataFrame to export
        powerbi_url (str): Power BI streaming dataset API endpoint
    """
    # Convert DataFrame to records (list of dicts)
    records = df.to_dict(orient='records')
    # Power BI expects a list of records in JSON
    payload = {"rows": records}
    response = requests.post(powerbi_url, json=payload)
    if response.status_code == 200:
        print("Data successfully sent to Power BI.")
    else:
        print(f"Failed to send data: {response.status_code} {response.text}")

# Example usage:
# df = pd.read_csv('sample_transactions.csv')
# export_to_powerbi(df, 'https://api.powerbi.com/beta/YOUR_WORKSPACE_ID/datasets/YOUR_DATASET_ID/rows?key=YOUR_KEY')
