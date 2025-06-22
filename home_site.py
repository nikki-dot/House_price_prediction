import streamlit as st
import pickle
import numpy as np

xgr = pickle.load(open("model.pkl",'rb'))
label = pickle.load(open("encoder.pkl",'rb'))

st.title("Predict Your House Price")
home = st.number_input('Home',min_value=0)
sqFt = st.number_input('SqFt',min_value=1000)
bedrooms = st.number_input('Bedrooms',min_value=0)
bathrooms = st.number_input('Bathrooms',min_value=0)
offers = st.number_input('Offers',min_value=0)
neighborhood = st.selectbox( "Neighborhood",['East','North','West'])
neighborhood_encoded = label.transform([neighborhood])[0]
house_price = (home,sqFt,bedrooms,bathrooms,offers,neighborhood_encoded)
house_price_array = np.asarray(house_price, dtype=np.float32).reshape(1,-1)
if st.button("Predicted Price"):
    predicted = xgr.predict(house_price_array)[0]
    st.success(f"Price is : ${predicted:.1f}")
