from alpha_vantage.fundamentaldata import FundamentalData
import streamlit as st
import pandas as pd
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = [s for s in sp500['Symbol'].unique().tolist() if s != 'VLTO']


ticker = st.selectbox("Select company for its fundamental data ", symbols_list)

key="WOF6G8PPXH8QI2EW"
st.title("Fundamental Data for "+ticker)
fd=FundamentalData(key,output_format='pandas')

st.subheader('company overview')
overview=fd.get_company_overview(ticker)[0]
ov=overview.T[2:]
ov.columns=list(overview.T.iloc[0])
st.write(ov)

# st.subheader('anual earnings')
# earnings=fd.get_anual_earnings(ticker)[0]
# ae=earnings.T[2:]
# ae.columns=list(earnings.T.iloc[0])
# st.write(ae)

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
