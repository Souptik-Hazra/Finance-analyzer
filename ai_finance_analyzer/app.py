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

df = None
if uploaded_file:
    df = load_data(uploaded_file)
    df = add_categories(df)
else:
    st.info("No file uploaded. You can generate synthetic data for demo.")
    if st.button("Generate Synthetic Data"):
        from synthetic_data_gan import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_samples=200)
        st.success("Synthetic data generated!")
        df.to_csv("synthetic_transactions.csv", index=False)
        st.write("Synthetic data saved as synthetic_transactions.csv")

if df is not None:
    st.subheader("Transactions")
    st.dataframe(df, use_container_width=True, hide_index=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Category Distribution")
        cat = df.groupby("Category")["Amount"].sum().reset_index()
        fig = px.pie(cat, names="Category", values="Amount", color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4)
        fig.update_traces(textinfo='percent+label', pull=[0.05]*len(cat))
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Expense Distribution")
        fig2 = px.histogram(df, x="Amount", nbins=20, color_discrete_sequence=["#636EFA"], marginal="box")
        fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0), bargap=0.2)
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        st.subheader("Monthly Trend")
        df['Month'] = df['Date'].dt.to_period('M')
        monthly = df.groupby('Month')['Amount'].sum().reset_index()
        monthly['Month'] = monthly['Month'].astype(str)
        fig3 = px.line(monthly, x='Month', y='Amount', markers=True, title='Monthly Expense Trend')
        st.plotly_chart(fig3, use_container_width=True)
    st.subheader("Anomaly Detection (Ensemble)")
    df = detect_anomalies(df)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Future Expense Prediction (Ensemble)")
    predictions = predict_future_expense(df)
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(y=predictions, mode='lines+markers', name='Predicted'))
    pred_fig.update_layout(title="Next 5 Predicted Expenses", xaxis_title="Future Days", yaxis_title="Amount", template="plotly_white")
    st.plotly_chart(pred_fig, use_container_width=True)
    st.subheader("AI Financial Insights")
    insights = generate_insights(df)
    for i in insights:
        st.success(i)
    st.subheader("Export Results")
    if st.button("Export to Excel"):
        df.to_excel("finance_results.xlsx", index=False)
        st.success("Exported to finance_results.xlsx")
    if st.button("Export to PDF"):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages("finance_results.pdf") as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.head(20).values, colLabels=df.columns, loc='center')
            pdf.savefig(fig)
        st.success("Exported to finance_results.pdf")
    # Placeholder for authentication and multi-user support
    st.sidebar.subheader("User Authentication (Coming Soon)")
    st.sidebar.info("Multi-user support will be added in future updates.")