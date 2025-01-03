import requests
import streamlit as st
import pandas as pd
# from alpha_vantage.alphaintelligence import  AlphaIntelligence
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = [s for s in sp500['Symbol'].unique().tolist() if s != 'VLTO']


ticker = st.selectbox("Select company to know news ", symbols_list)
st.title("news for "+ticker)

url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=10&apikey=WOF6G8PPXH8QI2EW'
r = requests.get(url)
data = r.json()

# st.write(data['feed'])
for i in range(0,10):
    st.subheader("Title")
    st.write(data['feed'][i]['title'])
    st.subheader("Summary")
    st.write(data['feed'][i]['summary'])
    st.subheader("Authors")
    if i < len(data['feed']) and i >= 0 and 'authors' in data['feed'][i] and len(data['feed'][i]['authors']) > 0:
        st.write(data['feed'][i]['authors'][0])
    else:
        st.write("alpha vantage")
    st.write("for more information click =>")
    st.write(data['feed'][i]['url'])
    st.write("***")
