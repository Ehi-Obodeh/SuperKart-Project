import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecasting", layout="centered")

st.title("SuperKart Sales Forecasting Dashboard")
st.write("Predict individual product sales or upload a batch CSV dataset for automated inference.")

# Navigation tabs or sections
tab1, tab2 = st.tabs(["Online Inference (Single)", "Batch Inference (CSV)"])

with tab1:
    st.header("Single Prediction Input")
    
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", value=12.66)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area", value=0.027)
        product_mrp = st.number_input("Product MRP", value=117.08)
        store_size = st.selectbox("Store Size", ["High", "Medium", "Low"])
    
    with col2:
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        product_id_char = st.text_input("Product Id Char", value="FD")
        store_age_years = st.number_input("Store Age (Years)", value=16)
        product_type_category = st.text_input("Product Type Category", value="Non Perishables")

    if st.button("Predict Sales Revenue"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category,
        }
        
        try:
            # Change URL to your deployed backend service URL or keep localhost for local Docker testing
            response = requests.post("http://localhost:7860/v1/predict", json=payload)
            if response.status_code == 200:
                prediction = response.json()["predicted_sales"]
                st.success(f"Predicted Sales Revenue: ${prediction:,.2f}")
            else:
                st.error(f"Backend error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend API: {e}")

with tab2:
    st.header("Batch Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        st.write("Uploaded File Preview:")
        import pandas as pd
        preview_df = pd.read_csv(uploaded_file)
        st.dataframe(preview_df.head())
        
        if st.button("Run Batch Predictions"):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://localhost:7860/v1/predictbatch", files=files)
                if response.status_code == 200:
                    results = response.json()
                    results_df = pd.DataFrame(list(results.items()), columns=["Index", "Predicted Sales"])
                    st.success("Batch predictions completed successfully!")
                    st.dataframe(results_df)
                else:
                    st.error(f"Batch processing error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}")
