def generate_insights(df):
    insights = []
    food = df[df['Category']=="Food"]['Amount'].sum()
    entertainment = df[df['Category']=="Entertainment"]['Amount'].sum()
    if food > 2000:
        insights.append("Your food spending is high this month.")
    if entertainment > 1500:
        insights.append("Entertainment spending increased.")
    max_expense = df.loc[df['Amount'].idxmax()]
    insights.append(f"Highest expense: {max_expense['Description']} ({max_expense['Amount']})")
    return insights