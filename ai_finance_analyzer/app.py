import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_processing import load_data, add_categories
from anomaly_detection import detect_anomalies
from prediction import predict_future_expense
from insights import generate_insights

st.set_page_config(page_title="AI Finance Analyzer", layout="wide", page_icon="💸", initial_sidebar_state="expanded")

# Custom Streamlit theme (PowerBI style)
st.markdown("""
    <style>
    .main {
        background-color: #f4f6fa;
    }
    .stApp {
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 1.5rem;
    }
    .st-bb, .st-cq, .st-dg {
        background: #fff;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💸 AI Personal Finance Analyzer")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df = add_categories(df)
    st.subheader("Transactions")
    st.dataframe(df, use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category Distribution")
        cat = df.groupby("Category")["Amount"].sum().reset_index()
        fig = px.pie(cat, names="Category", values="Amount", color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
        fig.update_traces(textinfo='percent+label', pull=[0.05]*len(cat))
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Expense Distribution")
        fig2 = px.histogram(df, x="Amount", nbins=10, color_discrete_sequence=["#636EFA"])
        fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0), bargap=0.2)
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Anomaly Detection")
    df = detect_anomalies(df)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Future Expense Prediction")
    predictions = predict_future_expense(df)
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(y=predictions, mode='lines+markers', name='Predicted'))
    pred_fig.update_layout(title="Next 5 Predicted Expenses", xaxis_title="Future Days", yaxis_title="Amount", template="plotly_white")
    st.plotly_chart(pred_fig, use_container_width=True)
    st.subheader("AI Financial Insights")
    insights = generate_insights(df)
    for i in insights:
        st.success(i)