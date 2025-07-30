
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import base64

# Load the trained model
model = joblib.load("model.pkl")

def predict_demand(data):
    data["Predicted_Demand"] = model.predict(data[["Past_Sales"]])
    data["Reorder_Point"] = (data["Predicted_Demand"] * 0.8).astype(int)
    data["Recommended_Stock"] = (data["Predicted_Demand"] * 1.2).astype(int)
    return data

def to_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Inventory Demand Prediction Report", ln=True, align="C")
    pdf.ln(10)

    col_names = df.columns.tolist()
    for col in col_names:
        pdf.cell(38, 10, col, 1, 0, 'C')
    pdf.ln()

    for index, row in df.iterrows():
        for item in row:
            pdf.cell(38, 10, str(item), 1, 0, 'C')
        pdf.ln()

    pdf.output("inventory_report.pdf")
    return "inventory_report.pdf"

def main():
    st.title("Retail Inventory Demand Predictor")
    st.write("Upload your sales data (CSV), and get predictions and reports.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Original Data:", df.head())

        predicted_df = predict_demand(df)
        st.write("Predicted Data:", predicted_df.head())

        st.markdown(to_csv_download_link(predicted_df, "predicted_inventory.csv"), unsafe_allow_html=True)

        if st.button("Generate PDF Report"):
            pdf_path = generate_pdf(predicted_df)
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="inventory_report.pdf">📄 Download PDF Report</a>'
            st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
