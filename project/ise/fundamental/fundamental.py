from alpha_vantage.fundamentaldata import FundamentalData
import streamlit as st
import pandas as pd

symbols_list = {
    "ADANIENTERPRISES": "ADANIENT.NS",
  "Asian Paints Ltd.": "ASIANPAINT.NS",
  "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
  "BAJAJFINSV": "BAJAJFINSV.NS",
  "BAJFINANCE": "BAJFINANCE.NS",
  "Bharti Airtel Ltd.": "BHARTIARTL.NS",
  "BPCL": "BPCL.NS",
  "Britannia Industries Ltd.": "BRITANNIA.NS",
  "CIPLA Ltd.": "CIPLA.NS",
  "डॉक्टर रेड्डीज लैबोरेटरीज (Dr. Reddys Laboratories Ltd.)": "DRREDDY.NS",
  "Eicher Motors Ltd.": "EICHERMOT.NS",
  "GAIL (India) Ltd.": "GAIL.NS",
  "Grasim Industries Ltd.": "GRASIM.NS",
  "HCL Technologies Ltd.": "HCLTECH.NS",
  "HDFC Bank Ltd.": "HDFC.NS",
  "Hero MotoCorp Ltd.": "HEROMOTOCO.NS",
  "Hindalco Industries Ltd.": "HINDALCO.NS",
  "Hindustan Unilever Ltd.": "HINDUNILVR.NS",
  "ITC Ltd.": "ITC.NS",
  "JSW Steel Ltd.": "JSWSTEEL.NS",
  "Kotak Mahindra Bank Ltd.": "KOTAKBANK.NS",
  "Larsen & Toubro Ltd.": "LT.NS",
  "LTIM Infra Ltd.": "LTIMINFRA.NS",
  "Mahindra & Mahindra Ltd.": "M&M.NS",
  "Maruti Suzuki India Ltd.": "MARUTI.NS",
  "Nestle India Ltd.": "NESTLE.NS",
  "NTPC Ltd.": "NTPC.NS",
  "Power Grid Corporation of India Ltd.": "POWERGRID.NS",
  "RELIANCE": "RELIANCE.NS",
  "DIVISLAB": "DIVISLAB.NS",  # Assuming this is the missing company
  "SHRI KAMADHENU": "SHKAMATADE.NS",  # Assuming this is the missing company
  "SBIN": "SBIN.NS",
  "State Bank of India": "SBIN.NS",  # Same ticker symbol as SBIN
  "Sun Pharmaceutical Industries Ltd.": "SUNPHARMA.NS",
  "TCS": "TCS.NS",
  "Tech Mahindra Ltd.": "TECHM.NS",
  "Titan Company Ltd.": "TITAN.NS",
  "UltraTech Cement Ltd.": "ULTRACEMCO.NS",
  "UPL Ltd.": "UPL.NS",
  "VEDL": "VEDL.NS",
  "Wipro Ltd.": "WIPRO.NS",
  "Zee Entertainment Enterprises Ltd.": "ZEEL.NS"
}



stock= st.selectbox("Select company for prediction ", symbols_list)
ticker=symbols_list[stock]


key="WRJ7G7AZ28PTJHFV"


def get_financial_data(ticker, function, delay=1):  # Function to handle API calls
    time.sleep(delay)  # Add a delay between requests (adjust as needed)
    url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={key}&outputsize=pandas"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"Error: {response.status_code} for function: {function}")
        return None  # Handle error by returning None

# st.title("Fundamental Data for " + ticker)



st.title("Fundamental Data for "+ticker)
fd=FundamentalData(key,output_format='pandas')
st.subheader('company overview')
overview=fd.get_company_overview(ticker)[0]
ov=overview.T[2:]
ov.columns=list(overview.T.iloc[0])
st.write(ov)

st.subheader('anual earnings')
earnings=fd.get_anual_earnings(ticker)[0]
ae=earnings.T[2:]
ae.columns=list(earnings.T.iloc[0])
st.write(ae)

st.subheader('balance sheet')
balance_sheet=fd.get_balance_sheet_annual(ticker)[0]
bs=balance_sheet.T[2:]
bs.columns=list(balance_sheet.T.iloc[0])
st.write(bs)

st.subheader('income Statement')
income_statement=fd.get_income_statement_annual(ticker)[0]
is1=income_statement.T[2:]
is1.columns=list(income_statement.T.iloc[0])
st.write(is1)

st.subheader('Cash flow statement')
cash_flow=fd.get_cash_flow_annual(ticker)[0]
cf=cash_flow.T[2:]
cf.columns=list(cash_flow.T.iloc[0])
st.write(cf)
