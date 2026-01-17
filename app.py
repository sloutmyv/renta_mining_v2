import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Evaluation rentabilité projet",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Evaluation rentabilité projet")

# Data fetching function
@st.cache_data
def get_btc_data(period_years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start_date, end=end_date)
    return df

try:
    # 1. Preliminary data fetch to get historical volatility
    current_years = st.session_state.get('years', 5)
    df = get_btc_data(current_years)
    
    if df.empty:
        st.error("Impossible de récupérer les données du BTC. Veuillez réessayer plus tard.")
        st.stop()
        
    prices = df['Close']
    returns = prices.pct_change().dropna()
    hist_vol_annual = float(returns.std() * np.sqrt(365) * 100)

    # 2. Parameters (always at the top)
    st.subheader("Paramètres de simulation")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)

    with p_col1:
        years = st.number_input("Période historique (années)", min_value=1, max_value=20, value=current_years, key='years')
    with p_col2:
        prediction_years = st.number_input("Période de prédiction (années)", min_value=1, max_value=10, value=2)
    with p_col3:
        n_simulations = st.number_input("Nombre de simulations", min_value=1, max_value=500, value=100)
    with p_col4:
        target_change_pct = st.number_input("Évolution cible (%)", min_value=-100.0, max_value=1000.0, value=5.0)
    with p_col5:
        user_volatility = st.number_input("Volatilité (annuelle %)", min_value=0.1, max_value=500.0, value=hist_vol_annual)

    # 3. Custom CSS for vertical centering of the metrics column
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
    # 4. Main layout columns (Chart + Metrics)
    main_col, side_metrics = st.columns([4, 1])

    # Simulation Logic
    volatility = user_volatility / 100 / np.sqrt(365)
    last_price = prices.iloc[-1]
    n_days = int(prediction_years * 365)
    
    target_ratio = 1 + (target_change_pct / 100)
    daily_drift = (np.log(target_ratio) / n_days) - (0.5 * (volatility**2))
    
    simulations = []
    future_dates = [prices.index[-1] + timedelta(days=i) for i in range(1, n_days + 1)]
    
    for i in range(n_simulations):
        random_returns = np.random.normal(daily_drift, volatility, n_days)
        price_path = last_price * np.exp(np.cumsum(random_returns))
        simulations.append(price_path)
        
    sim_array = np.array(simulations)
    sim_avg = np.mean(sim_array, axis=0)
    target_price = last_price * target_ratio
        
    # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        name="Historique BTC/USD",
        line=dict(color='royalblue', width=2)
    ))
    
    for i, path in enumerate(simulations):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=path,
            mode='lines',
            line=dict(color='rgba(0, 255, 0, 0.05)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=sim_avg,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)', width=3),
        name="Moyenne des simulations"
    ))
        
    fig.update_layout(
        title=f"Cours BTC/USD et Simulations Monte Carlo (+{target_change_pct}% cible)",
        xaxis_title="Date",
        yaxis_title="Prix (USD)",
        hovermode="x unified",
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    with main_col:
        st.plotly_chart(fig, use_container_width=True)
        
    with side_metrics:
        st.metric("Prix Actuel", f"{last_price:,.0f} $")
        st.metric("Prix Cible", f"{target_price:,.0f} $", delta=f"{target_change_pct}%")
        
except Exception as e:
    st.error(f"Une erreur est survenue : {e}")
