import streamlit as st
import matplotlib.pyplot as plt
from model import get_data, prepare_data, train_model, predict_future

st.title("📈 Stock Price Prediction App")

stock = st.text_input("Enter Stock Symbol (AAPL, TSLA, MSFT)")

if st.button("Predict"):
    df = get_data(stock)

    st.subheader("Stock Data")
    st.write(df.tail())

    X, y = prepare_data(df)
    model = train_model(X, y)

    predictions = predict_future(model, df)

    st.subheader("Predicted Prices (Next 30 Days)")
    st.write(predictions)

    plt.figure()
    plt.plot(predictions)
    st.pyplot(plt)