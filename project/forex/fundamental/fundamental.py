import streamlit as st

# Define a dictionary to store forex pair descriptions
currency_descriptions = {
    "EURUSD": "Euro (EUR) / US Dollar (USD): The most traded currency pair globally, reflecting the economic relationship between the Eurozone and the United States.",
    "USDJPY": "US Dollar (USD) / Japanese Yen (JPY): A major pair influenced by US economic data and Japanese monetary policy.",
    "GBPUSD": "British Pound (GBP) / US Dollar (USD): Represents the economic relationship between the UK and the US, impacted by factors like Brexit.",
    "AUDUSD": "Australian Dollar (AUD) / US Dollar (USD): Often nicknamed the 'Aussie dollar,' reflects the Australian economy and commodity prices.",
    "USDCAD": "US Dollar (USD) / Canadian Dollar (CAD): A major North American currency pair, influenced by factors like oil prices and interest rate differentials.",
    "USDCHF": "US Dollar (USD) / Swiss Franc (CHF): Considered a safe-haven pair, often sought during market volatility.",
    "NZDUSD": "New Zealand Dollar (NZD) / US Dollar (USD): The 'Kiwi dollar,' influenced by the New Zealand economy and agricultural exports.",
    "EURJPY": "Euro (EUR) / Japanese Yen (JPY): A cross currency pair reflecting the economic relationship between the Eurozone and Japan.",
    "GBPJPY": "British Pound (GBP) / Japanese Yen (JPY): Another cross currency pair, influenced by the UK and Japanese economies.",
    "EURGBP": "Euro (EUR) / British Pound (GBP): Represents the exchange rate between the Eurozone and the UK.",
    "AUDJPY": "Australian Dollar (AUD) / Japanese Yen (JPY): A cross currency pair, influenced by Australian commodity exports and Japanese monetary policy.",
    "EURAUD": "Euro (EUR) / Australian Dollar (AUD): Reflects the economic relationship between the Eurozone and Australia.",
    "EURCHF": "Euro (EUR) / Swiss Franc (CHF): Another safe-haven currency pair, often used during market uncertainty.",
    "AUDNZD": "Australian Dollar (AUD) / New Zealand Dollar (NZD): The 'Trans-Tasman' pair, reflecting economic ties between Australia and New Zealand.",
    "NZDJPY": "New Zealand Dollar (NZD) / Japanese Yen (JPY): A cross currency pair, influenced by the New Zealand economy and Japanese demand for NZ exports.",
    "GBPAUD": "British Pound (GBP) / Australian Dollar (AUD): Represents the exchange rate between the UK and Australia.",
    "GBPCAD": "British Pound (GBP) / Canadian Dollar (CAD): A less common major pair, influenced by economic ties between the UK and Canada.",
    "EURNZD": "Euro (EUR) / New Zealand Dollar (NZD): A cross currency pair, reflecting the economic relationship between the Eurozone and New Zealand.",
    "AUDCAD": "Australian Dollar (AUD) / Canadian Dollar (CAD): A minor pair, influenced by commodity prices and economic ties.",
    "GBPCHF": "British Pound (GBP) / Swiss Franc (CHF): Another less common pair, with safe-haven characteristics of the Swiss Franc.",
    "AUDCHF": "Australian Dollar (AUD) / Swiss Franc (CHF): A cross currency pair, influenced by Australian commodity exports and the Swiss Franc's safe-haven status.",
    "EURCAD": "Euro (EUR) / Canadian Dollar (CAD): A minor pair, reflecting economic ties between the Eurozone and Canada.",
    "CADJPY": "Canadian Dollar (CAD) / Japanese Yen (JPY): A less common cross currency pair, influenced by Canadian oil exports and Japanese demand.",
    "GBPNZD": "British Pound (GBP) / New Zealand Dollar (NZD): A minor cross currency pair, reflecting economic ties between the UK and New Zealand.",
    "CADCHF": "Canadian Dollar (CAD) / Swiss Franc (CHF): A minor pair, influenced by Canadian economic data and the safe-haven Swiss Franc.",
    "CHFJPY": "Swiss Franc (CHF) / Japanese Yen (JPY): A less common cross currency pair, with safe-haven characteristics of both currencies.",
    "NZDCAD": "New Zealand Dollar (NZD) / Canadian Dollar (CAD): A minor pair, reflecting economic ties between New Zealand and Canada.",
    "NZDCHF": "New Zealand Dollar (NZD) / Swiss Franc (CHF): A minor cross currency pair, influenced by the safe-haven Swiss Franc and NZ commodity exports."
}

# Set the application title
st.header("Fundamental data of Major Forex Pairs")
# st.subheader("")
# st.write("")
# st.write("***")
for n,d in currency_descriptions.items():
    # print(f"{state}: {capital}")
    st.subheader(f"{n}")
    st.write(f"{d}")
    st.write("***")


#
