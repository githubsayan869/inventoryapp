import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import base64
from io import BytesIO

# Load model
model = joblib.load("model.pkl")

def predict_demand(data, column):
    data["Predicted_Demand"] = model.predict(data[[column]])
    return data

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Retail Inventory Prediction Report", ln=True, align="C")
    pdf.ln(10)

    for i in range(min(10, len(data))):
        row = data.iloc[i]
        pdf.cell(200, 10, txt=f"Item {i+1} - {row.to_dict()}", ln=True)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

def get_pdf_download_link(pdf_data):
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="prediction_report.pdf">📄 Download Report as PDF</a>'
    return href

def main():
    st.title("Retail Inventory Optimization")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        column = st.selectbox("Select the column with sales data", df.columns)

        if st.button("Predict Demand"):
            predicted_df = predict_demand(df.copy(), column)

            st.success("Prediction Complete!")
            st.write(predicted_df.head())

            # CSV download
            csv = predicted_df.to_csv(index=False).encode()
            st.download_button("📥 Download Predicted CSV", csv, "predicted_data.csv", "text/csv")

            # PDF Report download
            pdf = generate_pdf(predicted_df)
            st.markdown(get_pdf_download_link(pdf), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
