### importing libraries


import streamlit as st
import pandas as pd 
import numpy as np
import yfinance as yf

import plotly.express as px
import datetime as dt





st.set_page_config(page_title = 'Market Insight Analysis using CAPM Model',
                   page_icon ="chart_with_upwards_trend",
                   layout= 'wide')

## adding page title to web application

st.title("Capital Assest Pricing Model")


#getting input from user


col1,col2 = st.columns([3,1])
with col1:
     stocks_list= st.multiselect(" ### Choose  stocks",('ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 
		'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 
		'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 
		'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'), 
	    ['ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS','JSWSTEEL.NS'])
with col2:
     year= st.number_input("Number of years", 1, 10)


#downloading data for SP500
     
  # Define the start and end dates which will be used to get past 10 years of data from today's date
start_date = dt.date(dt.date.today().year - year, dt.date.today().month, dt.date.today().day)
end_date = dt.date.today() 


## try:
     
     #fetching the ninfty 50 data from yahoo finance and storing as series
nifty50_data= yf.download('^NSEI', start= start_date, end= end_date)

     ## get date as column so reset the value 
nifty50_data.reset_index(inplace=True)
     
     # Get 'Date' and 'Close' columns only
data1= nifty50_data[['Date', 'Close']]
nifty50 = data1.copy()

	# Changing name of 'Close' column to 'Nifty50'
nifty50.rename(columns={'Close': 'Nifty50'}, inplace=True)


     ## feth the close price from the selected stocks to store as a dataframe

stock_df = pd.DataFrame()

for stock in stocks_list:
     data = yf.download(stock, period=f'{year}y')
     stock_df[f'{stock}'] = data['Close']

     
stock_df.reset_index(inplace=True)



     ## merge the stock_df and sp500 together to get the final dataframe for analysis 

stock_df =pd.merge(stock_df, nifty50, on='Date', how='inner')  

     ## displays stocks on web application 

col3, col4 = st.columns([2,2]) 
with col3:
          st.markdown("### Stocks Data Head ")
          st.dataframe(stock_df.head(), use_container_width= True )
with col4:
	     st.markdown("### Stocks Data Tail")
	     st.dataframe(stock_df.tail(), use_container_width=True)
             

# Function to plot interactive plotly line charts for selected stocks
def plot(df):
		fig = px.line()

		for i in df.columns[1:]:
			fig.add_scatter(x=df['Date'], y=df[i], name=i)
		fig.update_layout(width=700, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right',x=1))

		return fig


# Function to min-max normalization of the stock prices
def normalize(df2):
	    df = df2.copy()

	    for i in df.columns[1:]:
	        min_val = df[i].min()
	        max_val = df[i].max()
	        df[i] = (df[i] - min_val) / (max_val - min_val)

	    return df

col5, col6 = st.columns([1,1])
with col5:
	   st.markdown('### Stock Price Trends over Time')
	   st.plotly_chart(plot(stock_df), use_container_width=True)
with col6:
	   st.markdown('### Stock Price Trends ster Normalization')
	   st.plotly_chart(plot(normalize(stock_df)), use_container_width=True)



## functions to calculate daily returns
def stocks_daily_return(df):
	    df_daily_return = df.copy()
		## cpy of df is ued so that the changes that takes place in df does m=not affect the original df
		
	    
	    for i in df.columns[1:]:
	        df_daily_return[i] = ((df[i] - df[i].shift(1)) / df[i].shift(1)) * 100

	    return df_daily_return
# Calculate Daily Return for stocks and market(NIFTY50) over time

daily_return = stocks_daily_return(stock_df).dropna()

market_return =daily_return['Nifty50']

## covariance matrix for two different stock items


cov_matrix = daily_return.iloc[:,1:].cov()

market_variance = np.var(market_return)

## to store beta value for each stock
beta_value={}

for column in daily_return.iloc[:,1:].columns:
	covariance = cov_matrix[column]['Nifty50']
	beta = covariance / market_variance
	beta_value[column] = beta

# Calculating market portfolio return (rm)
market_portfolio_return = market_return.mean() * 252

# Considering Risk-Free Rate of Return (rf) as 0
risk_free_rate = 0.068

# Calculate Expected Return (ri) using CAPM
expected_return = {}
for stock, beta in beta_value.items():
		expected_returns = risk_free_rate + beta * (market_portfolio_return - risk_free_rate)
		expected_return[stock] = expected_returns

# Converting dict to dataframes
beta_df = pd.DataFrame.from_dict(beta_value, orient='index', columns=['Beta Value'])
beta_df.reset_index(inplace=True)
beta_df.rename(columns={'index': 'Stocks'}, inplace=True)

expected_return_df = pd.DataFrame.from_dict(expected_return, orient='index', columns=['Expected Return (in %)'])
expected_return_df.reset_index(inplace=True)
expected_return_df.rename(columns={'index': 'Stocks'}, inplace=True)


## displaying the final capm model

col7, col8 = st.columns([1, 1])
with col7:
		data2 = daily_return.iloc[:, :-1]
		st.markdown("### Daily Returns for Different Stocks")
		st.dataframe(data2.sort_values(by='Date', ascending=False).head(), use_container_width = True)
with col8:
		st.markdown("### Market Returns for NIFTY50")
		st.dataframe(daily_return[['Date', 'Nifty50']].sort_values(by='Date', ascending=False).head(), use_container_width=True)



## Function to plot bar chart for daily returns for different stocks
def plot2(df, col):
		color ='#0072BB'
		fig2 = px.bar(df, x=df['Date'], y=df[col], color_discrete_sequence=[color])
		fig2.update_layout(width=700, margin=dict(l=20, r=20, t=50, b=20), xaxis_title='', yaxis_title='')
		
		return fig2

	# Function to plot bar chart for market return for NIFTY50
def plot3(df):
		color = '#004B8D'
		fig3 = px.bar(df, x=df['Date'], y=df['Nifty50'], color_discrete_sequence=[color])
		fig3.update_layout(height=500, width=700, margin=dict(l=20, r=20, t=50, b=20), xaxis_title='', yaxis_title='')

		return fig3	

col9, col10 = st.columns([1, 1])
with col9:
		st.markdown("### Daily Return Trends")
		stock_lst = st.selectbox("Choose a stock", (stocks_list))
		st.plotly_chart(plot2(daily_return.iloc[:, :-1], stock_lst), use_container_width=True)
with col10:
		st.markdown("### Market Return Trends")
		st.plotly_chart(plot3(daily_return), use_container_width=True)
		

col11, col12 = st.columns([1, 1])
with col11:
		st.markdown("### Calculated Beta Value")
		st.dataframe(beta_df, use_container_width = True)
with col12:
		st.markdown("### Calculated Expected Returns using CAPM")
		st.dataframe(expected_return_df, use_container_width=True)
		

# Function to plot Bar chart for beta values of different stocks
		
         
def plot4(df):
		color = ['#9fc5e8', '#b4a7d6', '#e06666', '#93cf7a','#d5a6bd','#b496bd']
		fig4 = px.bar(x=df['Stocks'], y=df['Beta Value'],color_discrete_sequence=[color])
		fig4.update_layout(width=700, xaxis_title='stock', yaxis_title='beta value')
          
		return fig4

	# Function to plot Bar chart for expected returns of different stocks
def plot5(df):
		color = ['#dfa6ed','#a53170', '#bb3cbb' ,' #5f72dc' ,' #26b518' ,'#9584b1']
		fig5 = px.bar(x=df['Stocks'], y=df['Expected Return (in %)'], color_discrete_sequence=[color])
		fig5.update_layout(width=700, xaxis_title='stocks', yaxis_title='Expected return')

		return fig5

col13, col14 = st.columns([1, 1])
with col13:
		st.markdown("### Beta Values for Different Stocks and Market")
		st.plotly_chart(plot4(beta_df), use_container_width=True)
with col14:
		st.markdown("### Expected Returns for Different Stocks and Market")
		st.plotly_chart(plot5(expected_return_df), use_container_width=True)


# Calculating Expected Return for the portfolio
n = len(stock_df.columns) - 1
portfolio_weights = 1/n * np.ones(n)
er_portfolio = sum(np.array(list(expected_return.values())) * portfolio_weights)

st.subheader(f'Conclusion: The Expected Return Based on CAPM for the portfolio is roughly {round(er_portfolio, 2)}%')
##except:

	   ##st.write("Please select valid stock or refresh the web page...")