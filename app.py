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

@st.cache_data
def get_xpf_rate():
    try:
        ticker = yf.Ticker("USDXPF=X")
        rate = ticker.history(period="1d")['Close'].iloc[-1]
        return float(rate)
    except:
        return 110.0 # Fallback rate

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
    xpf_rate = get_xpf_rate()
    
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
    # --- Hardware & Mining Parameters ---
    st.subheader("Paramètres Projet & Matériel")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6, m_col7 = st.columns(7)
    with m_col1:
        n_machines = st.number_input("Machines", min_value=1, value=285)
    with m_col2:
        hashrate_unit_th = st.number_input("TH/s unitaire", min_value=1.0, value=200.0)
    with m_col3:
        power_unit_w = st.number_input("Watts unitaire", min_value=1, value=3500)
    with m_col4:
        tx_fees_block = st.number_input("Fees/bloc (BTC)", min_value=0.0, value=0.2, step=0.1)
    with m_col5:
        pool_fee_pct = st.number_input("Pool Fees (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    with m_col6:
        reject_rate_pct = st.number_input("Rejet (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
    with m_col7:
        start_date = st.date_input("Début projet", value=datetime(2025, 1, 1))

    st.subheader("Paramètres OPEX (Électricité & Uptime)")
    o_col1, o_col2, o_col3 = st.columns(3)
    with o_col1:
        cost_kwh_xpf = st.number_input("Coût Élec (XPF/kWh)", min_value=0.0, value=5.0, step=0.5)
    with o_col2:
        uptime_pct = st.slider("Uptime (Disponibilité %)", 0.0, 100.0, 98.0)
    with o_col3:
        pue_ratio = st.number_input("PUE (Refroidissement/Efficience)", min_value=1.0, value=1.1, step=0.05)

    st.subheader("Paramètres CAPEX (Investissement Initial en EUR)")
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        capex_machines_eur = st.number_input("CAPEX Machines (€)", min_value=0, value=975000)
    with c_col2:
        capex_infra_eur = st.number_input("CAPEX Infrastructure (€)", min_value=0, value=215000)
    with c_col3:
        capex_eng_eur = st.number_input("CAPEX Ingénierie & Divers (€)", min_value=0, value=55000)

    st.subheader("Paramètres OPEX Fixes (Mensuels en EUR)")
    of_col1, of_col2, of_col3, of_col4, of_col5, of_col6 = st.columns(6)
    with of_col1:
        opex_maint_eur = st.number_input("Maintenance (€)", min_value=0, value=2500)
    with of_col2:
        opex_super_eur = st.number_input("Supervision (€)", min_value=0, value=2000)
    with of_col3:
        opex_insur_eur = st.number_input("Assurance (€)", min_value=0, value=1000)
    with of_col4:
        opex_consum_eur = st.number_input("Consommables (€)", min_value=0, value=400)
    with of_col5:
        opex_net_eur = st.number_input("Réseau/Séc. (€)", min_value=0, value=200)
    with of_col6:
        opex_admin_eur = st.number_input("Admin/Compta (€)", min_value=0, value=500)

    # Constants
    EUR_XPF = 119.33174
    
    total_capex_xpf = (capex_machines_eur + capex_infra_eur + capex_eng_eur) * EUR_XPF
    monthly_fixed_opex_xpf = (opex_maint_eur + opex_super_eur + opex_insur_eur + opex_consum_eur + opex_net_eur + opex_admin_eur) * EUR_XPF
    daily_fixed_opex_xpf = monthly_fixed_opex_xpf / 30.44

    # Convert start_date to datetime64[ns, UTC] for comparison
    start_date_ts = pd.Timestamp(start_date).tz_localize('UTC')

    # Toggle for Daily vs Cumulative (Default: Cumulé)
    st.markdown("---")
    rev_mode = st.radio("Mode d'affichage des revenus", ["Journalier", "Cumulé"], index=1, horizontal=True)

    # Simulation Logic
    hashrate_total_h = n_machines * hashrate_unit_th * 1e12
    power_total_kw = (n_machines * power_unit_w) / 1000
    power_net_kw = power_total_kw * pue_ratio
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

    # Calculate Mining Revenue (BTC and USD)
    # Align difficulty to price index for historical calculation
    aligned_diff = diff_series.reindex(prices.index, method='ffill')
    hist_revenue_btc = (hashrate_total_h * 86400 * (hist_rewards + tx_fees_block)) / (aligned_diff * 2**32)
    hist_revenue_usd = hist_revenue_btc * prices
    
    rev_btc_sims = []
    rev_usd_sims = []
    for p_path, d_path in zip(price_sims, diff_sims):
        # future_rewards and d_path have the same length as future_dates
        r_btc = (hashrate_total_h * 86400 * (future_rewards.values + tx_fees_block)) / (np.array(d_path) * 2**32)
        rev_btc_sims.append(r_btc)
        rev_usd_sims.append(r_btc * p_path)
    
    rev_btc_avg = np.mean(np.array(rev_btc_sims), axis=0)
    rev_usd_avg = np.mean(np.array(rev_usd_sims), axis=0)

    # Mask and Prepare Revenue Display
    # Historical part filtered by start_date
    hist_mask = hist_revenue_btc.index >= start_date_ts
    hist_rev_filtered = hist_revenue_btc[hist_mask]
    
    # Simulation part
    sim_mask = [d >= start_date_ts for d in future_dates]
    future_dates_filtered = [d for d, m in zip(future_dates, sim_mask) if m]
    
    rev_paths_display = []
    for r_path in rev_btc_sims:
        # Combined filtered path
        path_filtered = [val for val, m in zip(r_path, sim_mask) if m]
        # Combine historical and simulation
        full_path = pd.Series(list(hist_rev_filtered.values) + list(path_filtered))
        
        if rev_mode == "Cumulé":
            rev_paths_display.append(full_path.cumsum())
        else:
            rev_paths_display.append(full_path)
            
    # X axis for full display
    full_dates_display = list(hist_rev_filtered.index) + list(future_dates_filtered)
    
    # Calculate average of paths
    rev_paths_array = np.array([p.values for p in rev_paths_display if len(p) > 0])
    if len(rev_paths_array) > 0:
        rev_avg_display = np.mean(rev_paths_array, axis=0)
    else:
        rev_avg_display = []

    # Calculate XPF Value for paths (Price * Path * Rate)
    # We need a price series that matches full_dates_display
    hist_prices_filtered = prices[hist_mask]
    
    rev_xpf_display = []
    for i, r_path in enumerate(rev_btc_sims):
        # r_path is just the prediction part
        path_filtered = [val for val, m in zip(r_path, sim_mask) if m]
        # p_path is the corresponding price path
        p_path = price_sims[i]
        p_path_filtered = [val for val, m in zip(p_path, sim_mask) if m]
        
        # Historical BTC * Hist Price + Prediction BTC * Pred Price
        # Then convert to XPF
        btc_values = list(hist_rev_filtered.values) + list(path_filtered)
        price_values = list(hist_prices_filtered.values) + list(p_path_filtered)
        
        daily_usd = np.array(btc_values) * np.array(price_values)
        if rev_mode == "Cumulé":
            rev_xpf_display.append(np.cumsum(daily_usd) * xpf_rate)
        else:
            rev_xpf_display.append(daily_usd * xpf_rate)

    rev_xpf_array = np.array(rev_xpf_display)
    if len(rev_xpf_array) > 0:
        rev_xpf_avg = np.mean(rev_xpf_array, axis=0)
    else:
        rev_xpf_avg = []

    # --- Gross Revenue Calculation (Applying Fees, Rejects, and Uptime) ---
    gross_factor_v1 = (1 - pool_fee_pct/100) * (1 - reject_rate_pct/100)
    # The user specifies that Gross Revenue should also take into account Uptime
    gross_factor = gross_factor_v1 * (uptime_pct / 100)
    
    rev_gross_btc_paths = [p * gross_factor for p in rev_paths_display]
    rev_gross_btc_avg = rev_avg_display * gross_factor if len(rev_avg_display) > 0 else []
    
    rev_gross_xpf_paths = [p * gross_factor for p in rev_xpf_display]
    rev_gross_xpf_avg = rev_xpf_avg * gross_factor if len(rev_xpf_avg) > 0 else []

    # --- Net Profit Calculation (Row 6) ---
    # Daily Electricity Cost in XPF
    # (Total kW) * 24h * Ckwh * PUE * Uptime/100
    daily_elec_cost_xpf = power_total_kw * 24 * cost_kwh_xpf * pue_ratio * (uptime_pct / 100)
    daily_total_opex_xpf = daily_elec_cost_xpf + daily_fixed_opex_xpf
    
    # We need to compute OPEX over the same periods (Historical + Simulation)
    n_total_days = len(full_dates_display)
    if rev_mode == "Cumulé":
        opex_cum_xpf = np.arange(1, n_total_days + 1) * daily_total_opex_xpf
        # Net Profit in XPF (Paths)
        net_profit_xpf_paths = [p - opex_cum_xpf for p in rev_gross_xpf_paths]
        net_profit_xpf_avg = rev_gross_xpf_avg - opex_cum_xpf if len(rev_gross_xpf_avg) > 0 else []
        
        # ROI Analysis (Row 7)
        # Starting from -CAPEX
        roi_xpf_paths = [p - total_capex_xpf for p in net_profit_xpf_paths]
        roi_xpf_avg = net_profit_xpf_avg - total_capex_xpf if len(net_profit_xpf_avg) > 0 else []
    else:
        # Daily View
        net_profit_xpf_paths = [p - daily_total_opex_xpf for p in rev_gross_xpf_paths]
        net_profit_xpf_avg = rev_gross_xpf_avg - daily_total_opex_xpf if len(rev_gross_xpf_avg) > 0 else []
        # ROI in daily mode is less intuitive, but we'll show cumulative ROI even in daily mode for row 7?
        # Actually, row 7 is usually cumulative. Let's force cumulative ROI in row 7.
        opex_cum_xpf_roi = np.arange(1, n_total_days + 1) * daily_total_opex_xpf
        # We need cumulative gross revenue even in daily mode for ROI
        roi_xpf_paths = []
        for p_daily in rev_gross_xpf_paths:
             roi_xpf_paths.append(np.cumsum(p_daily) - opex_cum_xpf_roi - total_capex_xpf)
        
        # Calculate gross avg cumulative
        rev_gross_xpf_avg_cum = np.cumsum(rev_gross_xpf_avg) if len(rev_gross_xpf_avg) > 0 else []
        roi_xpf_avg = rev_gross_xpf_avg_cum - opex_cum_xpf_roi - total_capex_xpf if len(rev_gross_xpf_avg_cum) > 0 else []

    total_opex_final = daily_total_opex_xpf * n_total_days if rev_mode == "Cumulé" else daily_total_opex_xpf
        
    # Plots (Standardized layout to align plotting areas)
    from plotly.subplots import make_subplots
    
    # 1. Price Chart
    fig_price = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Historique BTC/USD", line=dict(color='royalblue', width=2)), secondary_y=False)
    for i, p in enumerate(price_sims):
        fig_price.add_trace(go.Scatter(x=future_dates, y=p, mode='lines', line=dict(color='rgba(100, 149, 237, 0.15)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=False)
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
        fig_diff.add_trace(go.Scatter(x=future_dates, y=d, mode='lines', line=dict(color='rgba(255, 165, 0, 0.15)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=False)
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
        fig_reward.add_trace(go.Scatter(x=future_dates, y=r_usd, mode='lines', line=dict(color='rgba(0, 255, 255, 0.12)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=True)
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

    # --- RENDERING (Rows 1-3) ---
    st.markdown("---")
    price_col, price_metrics_col = st.columns([4, 1])
    with price_col: st.plotly_chart(fig_price, use_container_width=True)
    with price_metrics_col:
        st.metric("Prix Actuel", f"{last_price:,.0f} $")
        st.metric("Prix Cible", f"{target_price:,.0f} $", delta=f"{target_change_pct}%")

    st.markdown("---")
    diff_col, diff_metrics_col = st.columns([4, 1])
    with diff_col: st.plotly_chart(fig_diff, use_container_width=True)
    with diff_metrics_col:
        st.metric("Diff. Actuelle", f"{last_diff/1e12:,.0f}T")
        st.metric("Diff. Cible (moy)", f"{diff_avg[-1]/1e12:,.0f}T")

    st.markdown("---")
    reward_col, reward_metrics_col = st.columns([4, 1])
    with reward_col: st.plotly_chart(fig_reward, use_container_width=True)
    with reward_metrics_col:
        st.write("") # Placeholder

    # 4. Gross Revenue Chart (BTC Only)
    fig_rev = make_subplots(specs=[[{"secondary_y": True}]])
    # Main BTC trace
    for i, p in enumerate(rev_paths_display):
        fig_rev.add_trace(go.Scatter(x=full_dates_display, y=p.values, mode='lines', 
                                    line=dict(color='rgba(147, 112, 219, 0.12)', width=1), 
                                    showlegend=False, hoverinfo='skip'), secondary_y=False)
    
    if len(rev_avg_display) > 0:
        fig_rev.add_trace(go.Scatter(x=full_dates_display, y=rev_avg_display, 
                                    name=f"Moyenne {rev_mode} (BTC)", 
                                    line=dict(color='blueviolet', width=3)), secondary_y=False)

    # XPF traces (Ghost cloud + Avg)
    for p_xpf in rev_xpf_display:
        fig_rev.add_trace(go.Scatter(x=full_dates_display, y=p_xpf, mode='lines', 
                                    line=dict(color='rgba(0, 255, 127, 0.10)', width=1), 
                                    showlegend=False, hoverinfo='skip'), secondary_y=True)
    
    if len(rev_xpf_avg) > 0:
        fig_rev.add_trace(go.Scatter(x=full_dates_display, y=rev_xpf_avg, 
                                    name=f"Moyenne {rev_mode} (XPF)", 
                                    line=dict(color='springgreen', width=3)), secondary_y=True)

    # "Now" indicator
    fig_rev.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)
    
    title_suffix = " (Cumulé)" if rev_mode == "Cumulé" else " (Journalier)"
    fig_rev.update_layout(
        title=f"Revenu Théorique{title_suffix}",
        template="plotly_dark", height=450,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_rev.update_yaxes(title_text="BTC", secondary_y=False)
    fig_rev.update_yaxes(title_text="XPF", secondary_y=True)

    # Rendering row 4
    st.markdown("---")
    rev_row_col, rev_row_metrics_col = st.columns([4, 1])
    with rev_row_col: st.plotly_chart(fig_rev, use_container_width=True)
    with rev_row_metrics_col:
        # Metrics
        last_btc = rev_avg_display[-1] if len(rev_avg_display) > 0 else 0
        last_xpf = rev_xpf_avg[-1] if len(rev_xpf_avg) > 0 else 0
        label_suffix = "Total" if rev_mode == "Cumulé" else "Final"
        st.metric(f"{label_suffix} BTC", f"{last_btc:.4f} BTC")
        st.metric(f"{label_suffix} XPF", f"{last_xpf:,.0f} XPF")
        st.metric("Puissance Net (PUE incl.)", f"{power_net_kw:,.1f} kW")

    # 5. Gross Revenue Chart
    fig_gross = make_subplots(specs=[[{"secondary_y": True}]])
    # BTC traces
    for p in rev_gross_btc_paths:
        fig_gross.add_trace(go.Scatter(x=full_dates_display, y=p.values, mode='lines', 
                                     line=dict(color='rgba(255, 69, 0, 0.12)', width=1), 
                                     showlegend=False, hoverinfo='skip'), secondary_y=False)
    if len(rev_gross_btc_avg) > 0:
        fig_gross.add_trace(go.Scatter(x=full_dates_display, y=rev_gross_btc_avg, 
                                     name=f"Moyenne {rev_mode} (BTC)", 
                                     line=dict(color='orangered', width=3)), secondary_y=False)
    # XPF traces
    for p in rev_gross_xpf_paths:
        fig_gross.add_trace(go.Scatter(x=full_dates_display, y=p, mode='lines', 
                                     line=dict(color='rgba(255, 215, 0, 0.10)', width=1), 
                                     showlegend=False, hoverinfo='skip'), secondary_y=True)
    if len(rev_gross_xpf_avg) > 0:
        fig_gross.add_trace(go.Scatter(x=full_dates_display, y=rev_gross_xpf_avg, 
                                     name=f"Moyenne {rev_mode} (XPF)", 
                                     line=dict(color='gold', width=3)), secondary_y=True)

    fig_gross.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)
    fig_gross.update_layout(
        title=f"Revenu Brut (Net de frais Pool & Rejet){title_suffix}",
        template="plotly_dark", height=450,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_gross.update_yaxes(title_text="BTC", secondary_y=False)
    fig_gross.update_yaxes(title_text="XPF", secondary_y=True)

    # Rendering row 5
    st.markdown("---")
    gross_row_col, gross_row_metrics_col = st.columns([4, 1])
    with gross_row_col: st.plotly_chart(fig_gross, use_container_width=True)
    with gross_row_metrics_col:
        last_btc_g = rev_gross_btc_avg[-1] if len(rev_gross_btc_avg) > 0 else 0
        last_xpf_g = rev_gross_xpf_avg[-1] if len(rev_gross_xpf_avg) > 0 else 0
        st.metric(f"Brut {label_suffix} BTC", f"{last_btc_g:.4f} BTC")
        st.metric(f"Brut {label_suffix} XPF", f"{last_xpf_g:,.0f} XPF")

    # 6. Net Profit Chart (XPF focus)
    fig_net = make_subplots(specs=[[{"secondary_y": True}]])
    # XPF traces
    for p in net_profit_xpf_paths:
        fig_net.add_trace(go.Scatter(x=full_dates_display, y=p, mode='lines', 
                                   line=dict(color='rgba(0, 255, 0, 0.12)', width=1), 
                                   showlegend=False, hoverinfo='skip'), secondary_y=False)
    
    if len(net_profit_xpf_avg) > 0:
        # Color red if negative for the average path? Let's stay green for consistency but maybe bold
        fig_net.add_trace(go.Scatter(x=full_dates_display, y=net_profit_xpf_avg, 
                                   name=f"Profit Net Moy. (XPF)", 
                                   line=dict(color='#00FF00', width=3)), secondary_y=False)
    
    # Add a zero line
    fig_net.add_hline(y=0, line_width=1, line_color="white", opacity=0.3)
    fig_net.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)

    fig_net.update_layout(
        title=f"Profit Net (Après OPEX Élec){title_suffix}",
        template="plotly_dark", height=450,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_net.update_yaxes(title_text="XPF", secondary_y=False)
    # Ghost secondary axis for alignment
    fig_net.update_yaxes(showticklabels=False, secondary_y=True)

    # Rendering row 6
    st.markdown("---")
    net_row_col, net_row_metrics_col = st.columns([4, 1])
    with net_row_col: st.plotly_chart(fig_net, use_container_width=True)
    with net_row_metrics_col:
        st.metric("OPEX Élec", f"{total_opex_final:,.0f} XPF")
        st.metric("Puissance Net", f"{power_net_kw:,.1f} kW")

    # 7. ROI Chart
    fig_roi = make_subplots(specs=[[{"secondary_y": True}]])
    # ROI traces
    for p in roi_xpf_paths:
        fig_roi.add_trace(go.Scatter(x=full_dates_display, y=p, mode='lines', 
                                   line=dict(color='rgba(255, 255, 255, 0.08)', width=1), 
                                   showlegend=False, hoverinfo='skip'), secondary_y=False)
    
    if len(roi_xpf_avg) > 0:
        fig_roi.add_trace(go.Scatter(x=full_dates_display, y=roi_xpf_avg, 
                                   name="Flux de Trésorerie (Cumulé)", 
                                   line=dict(color='white', width=3)), secondary_y=False)
    
    # Add zero line (Break-even threshold)
    fig_roi.add_hline(y=0, line_width=2, line_color="lime", opacity=0.8, line_dash="solid")
    fig_roi.add_vline(x=today, line_width=1, line_dash="dash", line_color="white", opacity=0.5)

    fig_roi.update_layout(
        title="ROI & Flux de Trésorerie (Net de CAPEX & OPEX)",
        template="plotly_dark", height=450,
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    fig_roi.update_yaxes(title_text="XPF", secondary_y=False)
    fig_roi.update_yaxes(showticklabels=False, secondary_y=True)

    # Calculate Break-even Date
    break_even_date = "N/A"
    if len(roi_xpf_avg) > 0:
        be_indices = np.where(roi_xpf_avg >= 0)[0]
        if len(be_indices) > 0:
            break_even_date = full_dates_display[be_indices[0]].strftime("%Y-%m-%d")

    # Rendering row 7
    st.markdown("---")
    roi_row_col, roi_row_metrics_col = st.columns([4, 1])
    with roi_row_col: st.plotly_chart(fig_roi, use_container_width=True)
    with roi_row_metrics_col:
        last_roi_xpf = roi_xpf_avg[-1] if len(roi_xpf_avg) > 0 else 0
        st.metric("Point Mort Est.", break_even_date)
        st.metric("ROI Total (XPF)", f"{last_roi_xpf:,.0f} XPF")
        st.metric("CAPEX Total", f"{total_capex_xpf:,.0f} XPF")

    # 8. OPEX Breakdown (Pie Chart)
    monthly_elec_xpf = daily_elec_cost_xpf * 30.44
    opex_labels = ["Électricité", "Maintenance", "Supervision", "Assurance", "Consommables", "Réseau/Séc.", "Admin/Compta"]
    opex_values = [
        monthly_elec_xpf,
        opex_maint_eur * EUR_XPF,
        opex_super_eur * EUR_XPF,
        opex_insur_eur * EUR_XPF,
        opex_consum_eur * EUR_XPF,
        opex_net_eur * EUR_XPF,
        opex_admin_eur * EUR_XPF
    ]

    fig_pie = go.Figure(data=[go.Pie(labels=opex_labels, values=opex_values, hole=.3,
                                  marker=dict(colors=['#FF4B4B', '#FFA500', '#FFD700', '#00FA9A', '#1E90FF', '#9370DB', '#FF69B4']))])
    
    fig_pie.update_layout(
        title="Répartition des OPEX mensuels (XPF)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )

    st.markdown("---")
    pie_col1, pie_col2 = st.columns([1, 1])
    with pie_col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with pie_col2:
        total_monthly_opex = sum(opex_values)
        st.subheader("Résumé des charges mensuelles")
        st.write(f"**Total OPEX : {total_monthly_opex:,.0f} XPF / mois**")
        st.write(f"- Électricité : {monthly_elec_xpf:,.0f} XPF ({(monthly_elec_xpf/total_monthly_opex)*100:.1f}%)")
        st.write(f"- Charges Fixes : {monthly_fixed_opex_xpf:,.0f} XPF ({(monthly_fixed_opex_xpf/total_monthly_opex)*100:.1f}%)")
        
except Exception as e:
    import traceback
    st.error(f"Une erreur est survenue : {e}")
    st.code(traceback.format_exc())
