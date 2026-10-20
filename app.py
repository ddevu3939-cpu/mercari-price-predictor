
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

@st.cache_resource
def load_artifacts():
   model = joblib.load('lgb_model.pkl')
   return model

model = load_artifacts()

st.title("🛒 Mercari Price Suggestion App")
st.write("Enter product details below to get an estimated price prediction.")

with st.form("prediction_form"):
    item_name = st.text_input("Product Name", "Nike Air Max Running Shoes")
    brand_name = st.text_input("Brand Name", "Nike")
    category_name = st.selectbox("Category", ["Men/Shoes/Athletic", "Women/Athletic Apparel/Pants", "Electronics/Computers/Tablets", "Home/Kitchen/Dining"])
    item_condition_id = st.slider("Item Condition (1 = Best, 5 = Worst)", 1, 5, 1)
    shipping = st.selectbox("Shipping Paid By", options=[0, 1], format_func=lambda x: "Seller pays shipping" if x == 1 else "Buyer pays shipping")
    item_description = st.text_area("Product Description", "Brand new condition, worn only once.")
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    input_data = pd.DataFrame({
        'name': [item_name],
        'brand_name': [brand_name],
        'category_name': [category_name],
        'item_condition_id': [item_condition_id],
        'shipping': [shipping],
        'item_description': [item_description]
    })
    
    text_feat = vectorizer.transform(input_data['name'] + " " + input_data['item_description'])
    cat_feat = encoder.transform(input_data[['brand_name', 'item_condition_id', 'shipping']])
    X_input = hstack([text_feat, cat_feat]).tocsr()
    
    log_pred = model.predict(X_input)
    predicted_price = np.expm1(log_pred)[0]
    
    st.success(f"Estimated Product Price: **${predicted_price:.2f}**")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/lgb_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    encoder = joblib.load('models/encoder.pkl')
    return model, vectorizer, encoder

model, vectorizer, encoder = load_artifacts()

st.title("🛒 Mercari Price Suggestion App")
st.write("Enter product details below to get an estimated price prediction.")

with st.form("prediction_form"):
    item_name = st.text_input("Product Name", "Nike Air Max Running Shoes")
    brand_name = st.text_input("Brand Name", "Nike")
    category_name = st.selectbox("Category", ["Men/Shoes/Athletic", "Women/Athletic Apparel/Pants", "Electronics/Computers/Tablets", "Home/Kitchen/Dining"])
    item_condition_id = st.slider("Item Condition (1 = Best, 5 = Worst)", 1, 5, 1)
    shipping = st.selectbox("Shipping Paid By", options=[0, 1], format_func=lambda x: "Seller pays shipping" if x == 1 else "Buyer pays shipping")
    item_description = st.text_area("Product Description", "Brand new condition, worn only once.")
    
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    input_data = pd.DataFrame({
        'name': [item_name],
        'brand_name': [brand_name],
        'category_name': [category_name],
        'item_condition_id': [item_condition_id],
        'shipping': [shipping],
        'item_description': [item_description]
    })
    
    text_feat = vectorizer.transform(input_data['name'] + " " + input_data['item_description'])
    cat_feat = encoder.transform(input_data[['brand_name', 'item_condition_id', 'shipping']])
    X_input = hstack([text_feat, cat_feat]).tocsr()
    
    log_pred = model.predict(X_input)
    predicted_price = np.expm1(log_pred)[0]
    
    st.success(f"Estimated Product Price: **${predicted_price:.2f}**")

    
