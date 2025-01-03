import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date 
import datetime as dt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression


st.title("one day prediction")
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = [s for s in sp500['Symbol'].unique().tolist() if s != 'VLTO']

TODAY= date.today().strftime("%Y-%m-%d")
START = pd.to_datetime(TODAY)-pd.DateOffset(365*8)

selected_stock= st.selectbox("Select company for prediction ", symbols_list)
# Download the data
data = yf.download(selected_stock, START, TODAY)

# Create a DataFrame
data = pd.DataFrame(data)

# Reset the index
data.reset_index(inplace=True)
def MACD():
  def show_raw_data():
    st.write(data)
  show_raw_data()
  # Add Moving Average Convergence
  # /////////////////////////////////close
  data['m1']= data['Close'].ewm(span=12, adjust=False).mean() 
  data['m2']=data['Close'].ewm(span=26, adjust=False).mean()
  data['MACD1']=data['m1']-data['m2']
  data['Signal_Line1'] = data['MACD1'].ewm(span=9, adjust=False).mean()
  def predict_closing_price(data):
    last_macd = data['MACD1'][1]
    last_signal_line = data['Signal_Line1'][1]
    if last_macd > last_signal_line:
      return data['Close'].iloc[-1] + (last_macd - last_signal_line)
    else:
      return data['Close'].iloc[-1] - (last_macd - last_signal_line)

  predicted_closing_price = pd.Series(predict_closing_price(data))
  # , index=data['Close'].index[1:])
  max_date = max(data['Date'])
  extra_day = dt.timedelta(days=1)

  def plot_close_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    c_value = predicted_closing_price.iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m1'], name='12 days macd'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m2'], name='26 days macd'))

    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[c_value], mode='markers', marker=dict(color='red', size=10), name='closing Value', text=['closing Value: ' + str(c_value)], hoverinfo='text'))

    fig.layout.update(title_text="closing values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_close_data()

  # //////////////////OPEN/////////
  data['m3']=data['Open'].ewm(span=12, adjust=False).mean() 
  data['m4']=data['Open'].ewm(span=26, adjust=False).mean()
  data['MACD2'] = data['m3']-data['m4']
  data['Signal_Line2'] = data['MACD2'].ewm(span=9, adjust=False).mean()
  def predict_open_price(data):
    last_macd = data['MACD2'][1]
    last_signal_line = data['Signal_Line2'][1]
    if last_macd > last_signal_line:
      return data['Open'].iloc[-1] + (last_macd - last_signal_line)
    else:
      return data['Open'].iloc[-1] - (last_macd - last_signal_line)
  predicted_open_price = pd.Series(predict_open_price(data), index=data['Open'].index[1:])

  def plot_open_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    o_value = predicted_open_price.iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m3'], name='12 days macd'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m4'], name='26 days macd'))

    # Add a red point for the C value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[o_value], mode='markers', marker=dict(color='red', size=10), name='o Value', text=['o Value: ' + str(o_value)], hoverinfo='text'))

    fig.layout.update(title_text="opening values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_open_data()

# //// high///////////////////
  data['m5']=data['High'].ewm(span=12, adjust=False).mean() 
  data['m6']= data['High'].ewm(span=26, adjust=False).mean()
  data['MACD3'] = data['m5']-data['m6']
  data['Signal_Line3'] = data['MACD3'].ewm(span=9, adjust=False).mean()
  
  def predict_high_price(data):
    last_macd = data['MACD3'][1]
    last_signal_line = data['Signal_Line3'][1]
    if last_macd > last_signal_line:
      return data['High'].iloc[-1] + (last_macd - last_signal_line)
    else:
      return data['High'].iloc[-1] - (last_macd - last_signal_line)

  predicted_high_price = pd.Series(predict_high_price(data), index=data['High'].index[1:])
  def plot_high_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    h_value =predicted_high_price.iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='stock_high'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m5'], name='12 days macd'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m6'], name='26 days macd'))

    # Add a red point for the h value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[h_value], mode='markers', marker=dict(color='red', size=10), name='h Value', text=['h Value: ' + str(h_value)], hoverinfo='text'))

    fig.layout.update(title_text="High Values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_high_data()


# //////////////////////////////////////////////////////////////////low///////////////////////
  data['m7']= data['Low'].ewm(span=12, adjust=False).mean() 
  data['m8']=data['Low'].ewm(span=26, adjust=False).mean()
  data['MACD4'] =data['m7']-data['m8']
  data['Signal_Line4'] = data['MACD4'].ewm(span=9, adjust=False).mean()
  def predict_low_price(data):
    last_macd = data['MACD4'][1]
    last_signal_line = data['Signal_Line4'][1]
    if last_macd > last_signal_line:
      return data['Low'].iloc[-1] + (last_macd - last_signal_line)
    else:
      return data['Low'].iloc[-1] - (last_macd - last_signal_line)
  predicted_low_price = pd.Series(predict_low_price(data), index=data['Low'].index[1:])
  def plot_low_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    l_value = predicted_low_price.iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='stock_close'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m7'], name='12 days macd'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['m8'], name='26 days macd'))
    
    # Add a red point for the l value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[l_value], mode='markers', marker=dict(color='red', size=10), name='l Value', text=['l Value: ' + str(l_value)], hoverinfo='text'))

    fig.layout.update(title_text="Low Values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_low_data()

  container = st.container()
  col1, col2 = container.columns(2)


  # with col1:
  #   st.write("Predicted Closing Price:")
  #   st.write("+-",predicted_closing_price.iloc[0])

  #   st.write("Predicted High Price:+-")
  #   st.write("+-",predicted_high_price.iloc[0])

  # with col2:
  #   st.write("Predicted Opening Price:+-")
  #   st.write("+-",predicted_open_price.iloc[0])

  #   st.write("Predicted Low Price:+-")
  #   st.write("+-",predicted_low_price.iloc[0])

def LR():
  def show_raw_data():
    st.write(data)
  show_raw_data()
  
  closing_prices = data["Close"].values
  # Create the features and target variables
  X = np.arange(len(closing_prices)).reshape(-1, 1)
  y = closing_prices

# Create the linear regression model
  model = LinearRegression()

# Train the model
  model.fit(X, y)

# Make predictions for the next day
  next_day_prediction = model.predict([[len(closing_prices)]])

# Print the predicted closing price for the next day
  print(f"Predicted closing price for the next day: {next_day_prediction[0]:.2f}")

  max_date = max(data['Date'])
  extra_day = dt.timedelta(days=1)
  def plot_close_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    c_value = next_day_prediction[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))

    # Add a red point for the C value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[c_value], mode='markers', marker=dict(color='red', size=10), name='C Value', text=['C Value: ' + str(c_value)], hoverinfo='text'))

    fig.layout.update(title_text="Closing Values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_close_data()




  open_prices = data["Open"].values

# Create the features and target variables
  X = np.arange(len(open_prices)).reshape(-1, 1)
  y = open_prices

# Create the linear regression model
  model = LinearRegression()

# Train the model
  model.fit(X, y)

# Make predictions for the next day
  next_day_prediction = model.predict([[len(open_prices)]])

# Print the predicted closing price for the next day
  print(f"Predicted closing price for the next day: {next_day_prediction[0]:.2f}")

  def plot_open_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    o_value = next_day_prediction[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))

    # Add a red point for the C value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[o_value], mode='markers', marker=dict(color='red', size=10), name='o Value', text=['o Value: ' + str(o_value)], hoverinfo='text'))

    fig.layout.update(title_text="opening values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_open_data()


  high_prices = data["High"].values

# Create the features and target variables
  X = np.arange(len(high_prices)).reshape(-1, 1)
  y = high_prices

# Create the linear regression model
  model = LinearRegression()

# Train the model
  model.fit(X, y)

# Make predictions for the next day
  next_day_prediction = model.predict([[len(high_prices)]])

# Print the predicted closing price for the next day
  print(f"Predicted closing price for the next day: {next_day_prediction[0]:.2f}")

  def plot_high_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    h_value = next_day_prediction[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='stock_high'))

    # Add a red point for the h value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[h_value], mode='markers', marker=dict(color='red', size=10), name='h Value', text=['h Value: ' + str(h_value)], hoverinfo='text'))

    fig.layout.update(title_text="High Values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_high_data()


  low_prices = data["Low"].values

# Create the features and target variables
  X = np.arange(len(low_prices)).reshape(-1, 1)
  y = low_prices

# Create the linear regression model
  model = LinearRegression()

# Train the model
  model.fit(X, y)

# Make predictions for the next day
  next_day_prediction = model.predict([[len(low_prices)]])

# Print the predicted closing price for the next day
  print(f"Predicted closing price for the next day: {next_day_prediction[0]:.2f}")

  def plot_low_data():
    # Define c_value as the last value of the 'Close' column in the 'data' DataFrame
    l_value = next_day_prediction[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='stock_close'))

    # Add a red point for the C value
    fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1]+extra_day], y=[l_value], mode='markers', marker=dict(color='red', size=10), name='l Value', text=['l Value: ' + str(l_value)], hoverinfo='text'))

    fig.layout.update(title_text="Low Values", xaxis_title='Date', xaxis_range=[min(data['Date']), max_date + extra_day])
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)
  plot_low_data()

Methods=("LR","MACD")
selected_method= st.selectbox("Select method for prediction ", Methods)
if selected_method == "MACD":
    MACD()
if selected_method == "LR":
    LR()