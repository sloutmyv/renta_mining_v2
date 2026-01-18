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

# Data fetching functions
@st.cache_data
def get_btc_data(period_years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start_date, end=end_date)
    return df

@st.cache_data
def get_difficulty_data(period_years):
    url = f"https://api.blockchain.info/charts/difficulty?timespan={period_years}years&rollingAverage=24hours&format=json"
    import requests
    from datetime import timezone
    response = requests.get(url)
    data = response.json()
    # Ensure UTC timezone to match yfinance
    dates = [datetime.fromtimestamp(x['x'], tz=timezone.utc) for x in data['values']]
    values = [x['y'] for x in data['values']]
    return pd.Series(values, index=dates)

def get_block_reward(_date_series):
    from datetime import timezone
    # Standard Halving Dates (Approximate for blocks 210k, 420k, 630k, 840k, 1050k, 1260k)
    # All halvings are UTC for consistency
    halvings = [
        (datetime(2009, 1, 3, tzinfo=timezone.utc), 50.0),
        (datetime(2012, 11, 28, tzinfo=timezone.utc), 25.0),
        (datetime(2016, 7, 9, tzinfo=timezone.utc), 12.5),
        (datetime(2020, 5, 11, tzinfo=timezone.utc), 6.25),
        (datetime(2024, 4, 20, tzinfo=timezone.utc), 3.125),
        (datetime(2028, 3, 27, tzinfo=timezone.utc), 1.5625),
        (datetime(2032, 3, 1, tzinfo=timezone.utc), 0.78125),
    ]
    rewards = []
    for d in _date_series:
        current_reward = 50.0
        # Ensure 'd' is timezone aware for comparison
        d_utc = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        for h_date, h_val in halvings:
            if d_utc >= h_date:
                current_reward = h_val
        rewards.append(current_reward)
    return pd.Series(rewards, index=_date_series)

try:
    # 1. Preliminary data fetch
    current_years = st.session_state.get('years', 5)
    df_btc = get_btc_data(current_years)
    diff_series = get_difficulty_data(current_years)
    
    if df_btc.empty:
        st.error("Impossible de récupérer les données du BTC.")
        st.stop()
        
    prices = df_btc['Close']
    returns = prices.pct_change().dropna()
    hist_vol_annual = float(returns.std() * np.sqrt(365) * 100)
    
    # Calculate historical block rewards
    hist_rewards = get_block_reward(prices.index)

    # 2. Parameters
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

    st.subheader("Paramètres de Difficulté")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        sensitivity = st.slider("Sensibilité de la difficulté au prix", 0.0, 2.0, 0.7, help="Ratio de réaction de la difficulté aux variations du prix.")
    with d_col2:
        lag_days = st.slider("Inertie (jours de retard)", 0, 180, 60, help="Délai avant que les variations de prix n'impactent la difficulté. Pendant ce délai, les trajectoires suivent les prix historiques et restent donc groupées.")

    # 3. Custom CSS for layout refinements
    st.markdown("""
        <style>
        .stSubheader { margin-top: 2rem !important; }
        [data-testid="column"] { display: flex; flex-direction: column; justify-content: center; }
        </style>
        """, unsafe_allow_html=True)
        
    # 4. Main layout structure
    st.markdown("---")
    price_col, price_metrics_col = st.columns([4, 1])
    st.markdown("---")
    diff_col, diff_metrics_col = st.columns([4, 1])
    st.markdown("---")
    reward_col, reward_metrics_col = st.columns([4, 1])

    # Simulation Logic
    volatility = user_volatility / 100 / np.sqrt(365)
    last_price = prices.iloc[-1]
    last_diff = diff_series.iloc[-1]
    n_days = int(prediction_years * 365)
    
    target_ratio = 1 + (target_change_pct / 100)
    daily_drift = (np.log(target_ratio) / n_days) - (0.5 * (volatility**2))
    
    price_sims = []
    diff_sims = []
    
    # Universal future dates starting from the last price point (today)
    # This ensures all 3 charts' prediction zones start at the same X-coordinate
    today = prices.index[-1]
    future_dates = [today + timedelta(days=i) for i in range(0, n_days + 1)]
    
    # Block rewards for simulation
    future_rewards = get_block_reward(future_dates)
    
    for i in range(n_simulations):
        # Price path
        random_returns = np.random.normal(daily_drift, volatility, n_days)
        price_path = last_price * np.exp(np.cumsum(random_returns))
        full_sim_price_path = np.concatenate([[last_price], price_path])
        price_sims.append(full_sim_price_path)
        
        # Difficulty path
        full_price_for_diff = np.concatenate([prices.values, price_path])
        diff_path = [last_diff]
        # Adding a small intrinsic noise to difficulty so paths diverge immediately
        diff_noise = np.random.normal(0, 0.001, n_days) # 0.1% daily noise
        
        for t_idx in range(1, n_days + 1):
            price_idx = len(prices) + t_idx - lag_days - 1
            if price_idx >= 1:
                price_return = (full_price_for_diff[price_idx] / full_price_for_diff[price_idx - 1]) - 1
                # Difficulty changes based on lagged price return + intrinsic noise
                new_diff = diff_path[-1] * (1 + (price_return * sensitivity) + diff_noise[t_idx-1])
            else:
                new_diff = diff_path[-1] * (1 + diff_noise[t_idx-1])
            diff_path.append(new_diff)
        diff_sims.append(diff_path)
        
    price_avg = np.mean(np.array(price_sims), axis=0)
    diff_avg = np.mean(np.array(diff_sims), axis=0)
    target_price = last_price * target_ratio
    
    # Calculate Reward values in USD
    hist_rewards_usd = prices * hist_rewards
    reward_usd_sims = []
    for p_path in price_sims:
        reward_usd_sims.append(p_path * future_rewards.values)
    reward_usd_avg = np.mean(np.array(reward_usd_sims), axis=0)
        
    # Plots (Standardized layout to align plotting areas)
    from plotly.subplots import make_subplots
    
    # 1. Price Chart
    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Historique BTC/USD", line=dict(color='royalblue', width=2)), secondary_y=False)
    for i, p in enumerate(price_sims):
        fig_price.add_trace(go.Scatter(x=future_dates, y=p, mode='lines', line=dict(color='rgba(100, 149, 237, 0.05)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=False)
    fig_price.add_trace(go.Scatter(x=future_dates, y=price_avg, name="Moyenne des simulations", line=dict(color='deepskyblue', width=3)), secondary_y=False)
    
    # "Now" indicator
    fig_price.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)
    
    fig_price.update_layout(
        title=f"Cours BTC/USD (+{target_change_pct}% cible)",
        template="plotly_dark", height=400,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_price.update_yaxes(title_text="USD", secondary_y=False)
    fig_price.update_yaxes(showticklabels=False, secondary_y=True)

    # 2. Difficulty Chart
    fig_diff = make_subplots(specs=[[{"secondary_y": True}]])
    # Connect difficulty history precisely to simulation
    hist_diff_x = list(diff_series.index)
    hist_diff_y = list(diff_series.values)
    if hist_diff_x[-1] < today:
        hist_diff_x.append(today)
        hist_diff_y.append(hist_diff_y[-1])
        
    fig_diff.add_trace(go.Scatter(x=hist_diff_x, y=hist_diff_y, name="Difficulté Hist.", line=dict(color='orange', width=2)), secondary_y=False)
    for d in diff_sims:
        fig_diff.add_trace(go.Scatter(x=future_dates, y=d, mode='lines', line=dict(color='rgba(255, 165, 0, 0.05)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=False)
    fig_diff.add_trace(go.Scatter(x=future_dates, y=diff_avg, name="Diff. Moyenne", line=dict(color='gold', width=3)), secondary_y=False)
    
    # "Now" indicator
    fig_diff.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)
    
    fig_diff.update_layout(
        title="Difficulté de Minage",
        template="plotly_dark", height=400,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_diff.update_yaxes(title_text="Difficulté", secondary_y=False)
    fig_diff.update_yaxes(showticklabels=False, secondary_y=True)

    # 3. Reward Chart
    fig_reward = make_subplots(specs=[[{"secondary_y": True}]])
    fig_reward.add_trace(go.Scatter(x=hist_rewards.index, y=hist_rewards.values, name="Récompense Hist. (BTC)", line=dict(color='firebrick', width=2, shape='hv')), secondary_y=False)
    fig_reward.add_trace(go.Scatter(x=future_dates, y=future_rewards.values, name="Prédiction Récompense (BTC)", line=dict(color='red', width=2, dash="dash", shape='hv')), secondary_y=False)
    fig_reward.add_trace(go.Scatter(x=prices.index, y=hist_rewards_usd, name="Valeur Hist. (USD)", line=dict(color='rgba(0, 255, 255, 0.5)', width=2)), secondary_y=True)
    for r_usd in reward_usd_sims:
        fig_reward.add_trace(go.Scatter(x=future_dates, y=r_usd, mode='lines', line=dict(color='rgba(0, 255, 255, 0.03)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=True)
    fig_reward.add_trace(go.Scatter(x=future_dates, y=reward_usd_avg, name="Valeur Moyenne (USD)", line=dict(color='cyan', width=3)), secondary_y=True)
    
    # "Now" indicator
    fig_reward.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)

    fig_reward.update_layout(
        title="Récompense par Bloc (BTC & USD)",
        template="plotly_dark", height=400,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_reward.update_yaxes(title_text="BTC", secondary_y=False)
    fig_reward.update_yaxes(title_text="USD", secondary_y=True)

    # Rendering
    with price_col: st.plotly_chart(fig_price, use_container_width=True)
    with price_metrics_col:
        st.metric("Prix Actuel", f"{last_price:,.0f} $")
        st.metric("Prix Cible", f"{target_price:,.0f} $", delta=f"{target_change_pct}%")

    with diff_col: st.plotly_chart(fig_diff, use_container_width=True)
    with diff_metrics_col:
        st.metric("Diff. Actuelle", f"{last_diff/1e12:,.0f}T")
        st.metric("Diff. Cible (moy)", f"{diff_avg[-1]/1e12:,.0f}T")

    with reward_col: st.plotly_chart(fig_reward, use_container_width=True)
    with reward_metrics_col:
        st.write("") # Placeholder to maintain alignment with other rows
        
except Exception as e:
    st.error(f"Une erreur est survenue : {e}")
