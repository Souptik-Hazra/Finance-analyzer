import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, add_categories
from anomaly_detection import detect_anomalies
from prediction import predict_future_expense
from insights import generate_insights

st.set_page_config(page_title="AI Finance Analyzer", layout="wide")
st.title("AI Personal Finance Analyzer")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df = add_categories(df)
    st.subheader("Transactions")
    st.dataframe(df)
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Category Distribution")
        cat = df.groupby("Category")["Amount"].sum()
        fig,ax = plt.subplots()
        ax.pie(cat,labels=cat.index,autopct='%1.1f%%')
        st.pyplot(fig)
    with col2:
        st.subheader("Expense Distribution")
        fig2,ax2 = plt.subplots()
        sns.histplot(df["Amount"],kde=True,ax=ax2)
        st.pyplot(fig2)
    st.subheader("Anomaly Detection")
    df = detect_anomalies(df)
    st.dataframe(df)
    st.subheader("Future Expense Prediction")
    predictions = predict_future_expense(df)
    st.write("Next 5 predicted expenses")
    st.write(predictions)
    st.subheader("AI Financial Insights")
    insights = generate_insights(df)
    for i in insights:
        st.write("•",i)