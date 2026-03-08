import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def categorize_expense(description):
    description = description.lower()
    if "swiggy" in description or "restaurant" in description or "groceries" in description:
        return "Food"
    if "uber" in description or "petrol" in description:
        return "Transport"
    if "amazon" in description or "shopping" in description:
        return "Shopping"
    if "netflix" in description or "movie" in description:
        return "Entertainment"
    if "bill" in description:
        return "Bills"
    if "salary" in description:
        return "Income"
    return "Other"

def add_categories(df):
    df["Category"] = df["Description"].apply(categorize_expense)
    return df