def generate_insights(df):
    insights = []
    food = df[df['Category']=="Food"]['Amount'].sum()
    entertainment = df[df['Category']=="Entertainment"]['Amount'].sum()
    shopping = df[df['Category']=="Shopping"]['Amount'].sum()
    income = df[df['Category']=="Income"]['Amount'].sum()
    bills = df[df['Category']=="Bills"]['Amount'].sum()
    avg_expense = df['Amount'].mean()
    # Trend analysis
    if 'Date' in df.columns:
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_food = df[df['Category']=="Food"].groupby('Month')['Amount'].sum()
        if len(monthly_food) > 1:
            last = monthly_food.iloc[-1]
            prev = monthly_food.iloc[-2]
            if last > prev * 1.2:
                insights.append("Food spending increased sharply compared to last month.")
            elif last < prev * 0.8:
                insights.append("Food spending dropped compared to last month.")
        monthly_ent = df[df['Category']=="Entertainment"].groupby('Month')['Amount'].sum()
        if len(monthly_ent) > 1:
            last = monthly_ent.iloc[-1]
            prev = monthly_ent.iloc[-2]
            if last > prev * 1.2:
                insights.append("Entertainment spending increased sharply compared to last month.")
            elif last < prev * 0.8:
                insights.append("Entertainment spending dropped compared to last month.")
    # Percentile-based high spending
    food_percentile = df[df['Category']=="Food"]['Amount'].quantile(0.75)
    if food > food_percentile:
        insights.append("Your food spending is above the 75th percentile.")
    shopping_percentile = df[df['Category']=="Shopping"]['Amount'].quantile(0.75)
    if shopping > shopping_percentile:
        insights.append("Shopping expenses are above the 75th percentile.")
    # Spike detection
    if df['Amount'].max() > avg_expense * 3:
        insights.append("There is a significant expense spike detected.")
    # Personalized tips
    if income > 0 and (food/income) > 0.3:
        insights.append("Consider reducing food expenses to save more.")
    if bills > income * 0.5:
        insights.append("Bills are consuming more than half your income.")
    if shopping > income * 0.2:
        insights.append("Shopping is a major part of your spending.")
    if income < bills:
        insights.append("Warning: Your bills exceed your income.")
    max_expense = df.loc[df['Amount'].idxmax()]
    min_expense = df.loc[df['Amount'].idxmin()]
    insights.append(f"Highest expense: {max_expense['Description']} ({max_expense['Amount']})")
    insights.append(f"Lowest expense: {min_expense['Description']} ({min_expense['Amount']})")
    insights.append(f"Average expense: {avg_expense:.2f}")
    return insights