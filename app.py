import streamlit as st
from scipy.stats import norm
import math
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import random
import string


# Fonction pour télécharger les données en temps réel
def fetch_real_time_price(stock_ticker1):
    stock_data = yf.download(stock_ticker1, period="1d")
    return stock_data['Close'].iloc[-1]

# Fonction pour calculer le prix d'une option européenne
def calculate_european_option(S, K, T, r, sigma, option_type):
    # Calculer d1 et d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Calculer le prix en fonction du type d'option
    if option_type == "Call":
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # "Put"
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price


st.set_page_config(page_title='Application Web de Simulation')
# Création des onglets
tab1, tab2= st.tabs(["Simulation Monte Carlo des prix futurs d'Apple Inc. (AAPL)",  "Simulation d'options Européenes(Put & Call)"])
# Initialisation de st.session_state si ce n'est pas déjà fait
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None
    # Define a function to fetch real-time data
def fetch_stock_data(ticker_symbol):
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    return df['Close'][-1]

with tab1:
    st.header('Simulation Monte Carlo des prix futurs Apple Inc. (AAPL)')

    # Widgets for Monte Carlo simulation
    start_date = st.date_input('Start Date', datetime(2017, 1, 3), key='mc_start_date')
    end_date = st.date_input('End Date', datetime(2017, 11, 20), key='mc_end_date')
    num_simulations = st.number_input('Number of Simulations', 100, 10000, 1000, key='mc_num_simulations')
    num_days = st.number_input('Days to Forecast', 10, 365, 252, key='mc_num_days')
    if st.button('Run Monte Carlo Simulation', key='mc_run_simulations'):
        # Fetching stock data
        prices = web.DataReader('AAPL', 'av-daily', start_date, end_date, api_key='YOUR_API_KEY')['close']
        last_price = prices[-1]
        returns = prices.pct_change()
        daily_vol = returns.std()

        simulation_df = pd.DataFrame()

        for x in range(num_simulations):
            count = 0
            price_series = [last_price]

            for y in range(num_days):
                if count == 251:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1

            simulation_df[x] = price_series

        # Plotting
        plt.figure(figsize=(10,5))
        plt.plot(simulation_df)
        plt.axhline(y = last_price, color = 'r', linestyle = '-')
        plt.title('Monte Carlo Simulation: AAPL')
        plt.xlabel('Day')
        plt.ylabel('Price')
        st.pyplot(plt)

with tab2:
    st.header("Simulation d'options Européenes(Put & Call)")
    # Input fields for the European options simulation
    stock_ticker = st.text_input("Enter Stock Ticker", "AAPL", key='option_ticker').upper()
    S = fetch_real_time_price(stock_ticker)
    K = st.number_input("Strike Price (K)", value=100.0, key='option_strike')
    T = st.number_input("Time to Expiration (T) in Years", value=1.0, key='option_time')
    r = st.number_input("Risk-Free Rate (r)", value=0.01, key='option_rate')
    sigma = st.number_input("Volatility (σ)", value=0.2, key='option_volatility')
    option_type = st.selectbox("Type of Option", ["Call", "Put"], key='option_type')

    # Calculate option price button
    if st.button(f"Calculate {option_type} Option Price", key='calculate_option'):
        # European option price calculation
        option_price = calculate_european_option(S, K, T, r, sigma, option_type)
        st.success(f"The price of the {option_type} option is: {option_price:.2f}")


