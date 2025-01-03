# import streamlit as st
import requests
from bs4 import BeautifulSoup

symbols_list =  {  
  "EURUSD",  # Euro/US Dollar
    "USDJPY",  # US Dollar/Japanese Yen
    "GBPUSD",  # British Pound/US Dollar
    "AUDUSD",  # Australian Dollar/US Dollar
    "USDCAD",  # US Dollar/Canadian Dollar
    "USDCHF",  # US Dollar/Swiss Franc
    "NZDUSD",  # New Zealand Dollar/US Dollar
    "EURJPY",  # Euro/Japanese Yen
    "GBPJPY",  # British Pound/Japanese Yen
    "EURGBP",  # Euro/British Pound
    "AUDJPY",  # Australian Dollar/Japanese Yen
    "EURAUD",  # Euro/Australian Dollar
    "EURCHF",  # Euro/Swiss Franc
    "AUDNZD",  # Australian Dollar/New Zealand Dollar
    "NZDJPY",  # New Zealand Dollar/Japanese Yen
    "GBPAUD",  # British Pound/Australian Dollar
    "GBPCAD",  # British Pound/Canadian Dollar
    "EURNZD",  # Euro/New Zealand Dollar
    "AUDCAD",  # Australian Dollar/Canadian Dollar
    "GBPCHF",  # British Pound/Swiss Franc
    "AUDCHF",  # Australian Dollar/Swiss Franc
    "EURCAD",  # Euro/Canadian Dollar
    "CADJPY",  # Canadian Dollar/Japanese Yen
    "GBPNZD",  # British Pound/New Zealand Dollar
    "CADCHF",  # Canadian Dollar/Swiss Franc
    "CHFJPY",  # Swiss Franc/Japanese Yen
    "NZDCAD",  # New Zealand Dollar/Canadian Dollar
    "NZDCHF"
}

# import requests

def scrape_dailyfx_news():
    url = "https://www.dailyfx.com/EURUSD/news-and-analysis"  # DailyFX EUR/USD news page URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    print(soup)
    # Find the relevant section containing news headlines
    news_headlines = soup.find_all("h2", class_="dfx-article-title")
    print(news_headlines)
    # Print the top 10 news headlines
    print("Top 10 news articles related to EUR/USD:")
    for i, headline in enumerate(news_headlines[:10], start=1):
        print(f"{i}. {headline.text.strip()}")

if __name__ == "__main__":
    scrape_dailyfx_news()






# stock= st.selectbox("Select company for prediction ", symbols_list)
# ticker=symbols_list[stock]

# #symbols_list[stock]
# # Replace 'YOUR_API_KEY' with your actual API key (if required)
# api_url = "https://newsapi.org/v2/everything"
# params = {
#     "q": {ticker},
#     "apiKey": "3245a94aa30c46db85423f56315e9048",  # Get an API key from newsapi.org
#     "pageSize": 10,
#     "language": "en",  # Number of articles to retrieve
# }

# response = requests.get(api_url, params=params)
# news_data = response.json()


# # print(news_data["articles"])
# # Print the top 10 news headlines

# print("Top 10 news articles related to SBI stock:")
# for article in news_data["articles"]:
#     st.subheader("Title")
#     st.write(article["title"])
#     st.subheader("Authors")
#     st.write(article["source"]["name"])
#     st.subheader("Summary")
#     st.write(article["description"])
#     st.write("for more information click =>")
#     st.write(article["url"])
#     st.write("***")
