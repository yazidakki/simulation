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
st.header('choisir une simulation svp')
tab1, tab2,tab3,tab4= st.tabs([" Monte Carlo ",  " Options Européenes(Put & Call)"," Mouvement brownien standard","Mouvement brownien geometrique"])
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
with tab3:
    # st.set_page_config(page_icon=":game_die:", page_title="Aboulaala Maria")
    # st.header(':one: Simulation du mouvement brownien standard')

    with st.expander("Introduction:"):
        
        st.markdown("""

        Un processus stochastique est une collection de variables aleatoires indicées {$W_t$}, ou $t \in T$  
        Un processus stochastique W : [0, +$\infty$[ x $\mathbb{R}$ $\longrightarrow$ $\mathbb{R}$ est mouvement brownien standard si: \n
        - $W_0$ = 0
        - Pour tout s$\leq$t , $W_t$ - $W_{t-1}$ suit la loi $\mathcal{N}$(0,t-s)
        - Pour tout n$\geq$1 , et tous $t_0$ = 0 < $t_1$ < ...< $t_n$, les accroissement ($W_{{t_i}+1}$ - $W_{t_i}$ : 0 $\leq$ i $\leq$ n-1) sont **independantes**.
        En d'autres termes, pour tout $t_0$, $W_t$ $\sim$ $\mathcal{N}$(0,t), les trajectoires de $W_t$ ,  $t_0$ sont presque surement continues.
                    """
        )

    st.subheader('Entrer le parametres de la simulation: :key:')
    with st.form(key="my_form"):
        d = st.number_input('Le nombre de simulation', step=1,min_value=1 )
        n = st.number_input('La periode', step=1, min_value=200)
        

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



    st.subheader("Mon code : :female-technologist: ")

    code = '''times = np.linspace(0. , T, n)
    dt = times[1] - times[0]
    dB = np.sqrt(dt)* np.random.normal(size=(n-1,d))
    B0 = np.zeros(shape=(1, d))
    B = np.concatenate((B0, np.cumsum(dB, axis=0)) , axis = 0)
    plt.plot(times, B)
    figure=plt.show()
    '''
    st.code(code, 

    language='python')




    st.markdown(
        """
    ---

    Realisé par Aboulaala Maria                  
    
        """
    )




#Soit ( $\Omega$, $\mathcal{F}$, $\mathbb{F}$, $\mathcal{P}$) un espace probabilisé filtré \n
with tab4:
    # st.set_page_config(page_icon="🐤", page_title="Aboulaala Maria")
    # st.header(':two: Simulation du mouvement brownien geometrique')

    with st.expander("Introduction:"):
        st.markdown("""
        Un processus stochastique {$X$($t$) : t$\geq$0 } est appele un mouvement brownien geometrique s'il satisfait l'equation differentielle stochastique  \n
        > d$S_t$ = $\mu$($S_t$)dt +   $\sigma$($S_t$)d$W_t$ \n
        où $W_t$ est un mouvement brownien standard,  $\mu$ $\in$ $\mathbb{R}$ et $\sigma$ > 0, De plus il est bien connu que la solution unique de cette equation est \n  
        > $S_t$ = $S_0$exp[($\mu$ - $\sigma^2$ /2)t + $\sigma$($W_t$)] \n
        

                
                    """)

    st.markdown("""

    > $S_t$ = $S_0$exp[($\mu$ - $\sigma^2$/2)t + $\sigma$($W_t$)]

                """)


    with st.form(key="my_form1"):
        mu = st.number_input('la derivé <mu>', step=0.1,min_value=0.1)
        sigma = st.number_input('la volatilité <sigma>', step=0.1, min_value=0.1)
        M = st.number_input('le nombre de simalation', step=1,min_value=1)
        S0 = st.number_input('Le prix initil du stock', step=1, min_value=1)
        n = st.number_input('La periode', step=1, min_value=50)
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


    st.subheader("Mon code : :female-technologist: ")

    code = '''
    dt = T/n
    St = np.exp(
        (mu - sigma ** 2 / 2 ) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size = (M,n)).T
    )
    St = np.vstack([np.ones(M), St])
    St = S0 * St.cumprod(axis=0)
    time = np.linspace(0, T, n+1)
    tt = np.full(shape=(M, n+1), fill_value=time).T
    plt.plot(tt, St)
    plt.show()
    '''
    st.code(code, 

    language='python')
    st.markdown(
        """
    ---

    Realisé par Aboulaala Maria                  

        """
    )
