
import requests
import streamlit as st
import pandas as pd
# from alpha_vantage.alphaintelligence import  AlphaIntelligence
symbols_list = ['BTC','ETH','BNB','SOL','XRP','DOGE']


ticker = 'CRYPTO:'+st.selectbox("Select cryptocurrency to know news ", symbols_list)
st.title("Fundamental Data for "+ticker)

url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=10&apikey=WOF6G8PPXH8QI2EW'

r = requests.get(url)
data = r.json()

# print(data)
# st.write(data)
for i in range(0,10):
    st.subheader("Title")
    # st.write(data['feed'][i]['title'])
    try:
        st.write(data['feed'][i]['title'])
  # ... rest of the code for displaying news info
    except KeyError:
        st.write("Error: News data not available")
    st.subheader("Summary")
    # st.write(data['feed'][i]['summary'])
    try:
        st.write(data['feed'][i]['title'])
  # ... rest of the code for displaying news info
    except KeyError:
        st.write("Error: News data not available")
    try:   
        st.subheader("Authors")
        if i < len(data['feed']) and i >= 0 and 'authors' in data['feed'][i] and len(data['feed'][i]['authors']) > 0:
            st.write(data['feed'][i]['authors'][0])
        else:
            st.write("alpha vantage")
        st.write("for more information click =>")
        st.write(data['feed'][i]['url'])
        st.write("***")
    except KeyError:
        st.write("cant be disclosed")
