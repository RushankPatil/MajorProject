import streamlit as st
import requests


symbols_list =  {  "ADANIENTERPRISES": "ADANIENT",
  "Asian Paints Ltd.": "ASIANPAINT",
  "BAJAJ-AUTO": "BAJAJ-AUTO",
  "BAJAJFINSV": "BAJAJFINSV",
  "BAJFINANCE": "BAJFINANCE",
  "Bharti Airtel Ltd.": "BHARTIARTL",
  "BPCL": "BPCL",
  "Britannia Industries Ltd.": "BRITANNIA",
  "CIPLA Ltd.": "CIPLA",
  "डॉक्टर रेड्डीज लैबोरेटरीज (Dr. Reddys Laboratories Ltd.)": "DRREDDY",
  "Eicher Motors Ltd.": "EICHERMOT",
  "GAIL (India) Ltd.": "GAIL",
  "Grasim Industries Ltd.": "GRASIM",
  "HCL Technologies Ltd.": "HCLTECH",
  "HDFC Bank Ltd.": "HDFC",
  "Hero MotoCorp Ltd.": "HEROMOTOCO",
  "Hindalco Industries Ltd.": "HINDALCO",
  "Hindustan Unilever Ltd.": "HINDUNILVR",
  "ITC Ltd.": "ITC",
  "JSW Steel Ltd.": "JSWSTEEL",
  "Kotak Mahindra Bank Ltd.": "KOTAKBANK",
  "Larsen & Toubro Ltd.": "LT",
  "LTIM Infra Ltd.": "LTIMINFRA",
  "Mahindra & Mahindra Ltd.": "M&M",
  "Maruti Suzuki India Ltd.": "MARUTI",
  "Nestle India Ltd.": "NESTLE",
  "NTPC Ltd.": "NTPC",
  "Power Grid Corporation of India Ltd.": "POWERGRID",
  "RELIANCE": "RELIANCE",
  "DIVISLAB": "DIVISLAB",  # Assuming this is the missing company
  "SHRI KAMADHENU": "SHKAMATADE",  # Assuming this is the missing company
  "SBIN": "SBIN",
  "State Bank of India": "SBIN",  # Same ticker symbol as SBIN
  "Sun Pharmaceutical Industries Ltd.": "SUNPHARMA",
  "TCS": "TCS",
  "Tech Mahindra Ltd.": "TECHM",
  "Titan Company Ltd.": "TITAN",
  "UltraTech Cement Ltd.": "ULTRACEMCO",
  "UPL Ltd.": "UPL",
  "VEDL": "VEDL",
  "Wipro Ltd.": "WIPRO",
  "Zee Entertainment Enterprises Ltd.": "ZEEL"
}

stock= st.selectbox("Select company for prediction ", symbols_list)
ticker=symbols_list[stock]

#symbols_list[stock]
# Replace 'YOUR_API_KEY' with your actual API key (if required)
api_url = "https://newsapi.org/v2/everything"
params = {
    "q": {ticker},
    "apiKey": "3245a94aa30c46db85423f56315e9048",  # Get an API key from newsapi.org
    "pageSize": 10,
    "language": "en",  # Number of articles to retrieve
}

response = requests.get(api_url, params=params)
news_data = response.json()


# print(news_data["articles"])
# Print the top 10 news headlines

print("Top 10 news articles related to SBI stock:")
for article in news_data["articles"]:
    st.subheader("Title")
    st.write(article["title"])
    st.subheader("Authors")
    st.write(article["source"]["name"])
    st.subheader("Summary")
    st.write(article["description"])
    st.write("for more information click =>")
    st.write(article["url"])
    st.write("***")
