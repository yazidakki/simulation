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
st.header('choisir une simulation svp💹')
tab1, tab2,tab3,tab4= st.tabs([" Mouvement brownien standard📊","Mouvement brownien geometrique📉"," Monte Carlo📉 ",  " Options Européenes(Put & Call)🪙"])
# Initialisation de st.session_state si ce n'est pas déjà fait
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None
    # Define a function to fetch real-time data
def fetch_stock_data(ticker_symbol):
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    return df['Close'][-1]

with tab3:
    st.header("Entrez les paramétres s'il vous plait::key:")

    # Widgets for Monte Carlo simulation
    
    num_simulations = st.slider('Nombre de simulation',min_value=1, max_value=1000, value=50, step=1 )

    num_days =  st.slider('jours de prévisions', min_value=0, max_value=100, value=50, step=1)
    start_date = st.date_input('date de début', datetime(2014, 1, 3), key='mc_start_date')
    end_date = st.date_input('date de fin', datetime(2014, 11, 20), key='mc_end_date')
    if st.button('cliquez(pour faire la simulation)', key='mc_run_simulations'):
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

with tab4:
    st.header("Entrez les paramétres s'il vous plait: :key:")
    # Input fields for the European options simulation
    stock_ticker = st.text_input("Enter Stock Ticker", "AAPL", key='option_ticker').upper()
    S = fetch_real_time_price(stock_ticker)
  
    K = st.slider("prix de l'exercice (K)", min_value=0, max_value=1000, value=10, step=1)
    T = st.number_input("Time to Expiration (T) in Years", value=1.0, key='option_time')
    r = st.number_input("taux sans risque(r)", value=0.01, key='option_rate')
    
    sigma = st.number_input("Volattilité (σ)", value=0.2, key='option_volatility')
    option_type = st.selectbox("Type de l'option", ["Call", "Put"], key='option_type')

    # Calculate option price button
    if st.button(f"Calculer le prix du {option_type} de l'Option ", key='calculate_option'):
        # European option price calculation
        option_price = calculate_european_option(S, K, T, r, sigma, option_type)
        st.success(f"Le prix du  {option_type} de l'option est: {option_price:.2f}")
st.markdown(
    """
---

 Realisé par YAZID AKKI                

    """
)
with tab1:
    
    # st.set_page_config(page_icon=":game_die:", page_title="Aboulaala Maria")
    # st.header(':one: Simulation du mouvement brownien standard')

   
        
        
    st.subheader('Entrer le parametres de la simulation: :key:')
    with st.form(key="my_form"):
        
        d = st.slider('Le nombre de simulation', min_value=1, max_value=1000, value=50, step=1 )
        n = st.slider('La periode', min_value=1, max_value=250, value=50, step=1)
        

        st.form_submit_button("Simuler")
    #nbr de simulation
    T=4

    times = np.linspace(0. , T, n)
    dt = times[1] - times[0]
    dB = np.sqrt(dt)* np.random.normal(size=(n-1,d))
    B0 = np.zeros(shape=(1, d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)) , axis = 0)
    #plt.plot(times, B)
    #figure=plt.show()

    st.set_option('deprecation.showPyplotGlobalUse', False)

    #st.pyplot(figure)

    st.subheader("La simulation : :star2: ")
    st.line_chart(B, use_container_width=True)
    st.subheader("Appercu des valeurs generées: :1234:")
    st.write(B)
    #st._arrow_line_chart(B)



    




#Soit ( $\Omega$, $\mathcal{F}$, $\mathbb{F}$, $\mathcal{P}$) un espace probabilisé filtré \n
with tab2:
    


    with st.form(key="my_form1"):
        mu = st.number_input('la derivé <mu>', step=0.1,min_value=0.1)
        sigma = st.slider('la volatilité <sigma>', step=0.1, min_value=0.1)
       
        M = st.slider('Le nombre de simulation', min_value=1, max_value=1000, value=50, step=1 )
        S0 = st.slider('Le prix initil du stock', min_value=1, max_value=1000, value=50, step=1)
        n = st.slider('La periode', min_value=1, max_value=250, value=50, step=1)
        st.form_submit_button("Simuler")




    T = 1


    dt = T/n
    #simulating using np array
    St = np.exp(
        (mu - sigma ** 2 / 2 ) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size = (M,n)).T
    )
    #imclude array of ones
    St = np.vstack([np.ones(M), St])

    #multiply bu S0 
    St = S0 * St.cumprod(axis=0)

    time = np.linspace(0, T, n+1)


    tt = np.full(shape=(M, n+1), fill_value=time).T

    #plt.plot(tt, St)
    #plt.show()


    st.subheader("Graphe generé :star2:")
    st.line_chart(St, use_container_width=True)

    st.subheader("Appercu des valeurs generé :1234:")

    st.write(St)



    