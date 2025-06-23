import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from portfolio import Portfolio
import os
import subprocess
import json

# ---------- Modern Styling ----------
st.set_page_config(page_title="S&P 500 AI Trader Dashboard", layout="wide")

st.markdown("""
    <style>
    body { background-color: #f9fbfc; }
    .block-container { padding: 2rem 2rem 3rem; }
    .stTabs [role="tablist"] { gap: 10px; }
    .stTabs [role="tab"] {
        font-weight: 600;
        border: 1px solid #e0e0e0;
        border-radius: 8px 8px 0 0;
        background-color: #fafafa;
        color: #333;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #0e76fd;
        color: #0e76fd;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f1f1f1 !important;
    }
    .stButton > button {
        border-radius: 6px;
        background-color: #0e76fd;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #105cd1;
    }
    .stMarkdown h3 { margin-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #0e76fd;'>📊 S&P 500 AI Trader Dashboard</h1>
<hr style='margin-top: -0.5rem; margin-bottom: 2rem; border: none; height: 1px; background: #e0e0e0;' />
""", unsafe_allow_html=True)

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Settings")
    view_mode = st.radio("View Mode", ["Trader", "Analyst"])

# ---------- Portfolio Setup ----------
portfolio_path = "data/history/portfolio.json"
try:
    with open(portfolio_path) as f:
        portfolio = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    portfolio = {}

# Ensure portfolio has all required keys
portfolio.setdefault("cash", 5000)
portfolio.setdefault("holdings", {})
portfolio.setdefault("positions", {})
portfolio.setdefault("history", {})

# ---------- Tabs ----------
tabs = st.tabs([
    "🧭 Overview", 
    "📈 Market View", 
    "📉 Stock Analyzer",
    "💼 Portfolio Tracker", 
    "🚨 Alerts & Sectors",
    "📊 Momentum Scanner", 
    "🧠 Explainability & Backtest"
])

# ---------- Tab 0: Overview ----------
with tabs[0]:
    st.subheader("🧭 AI Market Overview")

    import subprocess
    import sys

    # --- Prediction Trigger ---
    st.markdown("### 🔮 Generate Latest Predictions")
    if st.button("Run predict_all.py"):
        with st.spinner("Running prediction model..."):
            try:
                result = subprocess.run(
                    [sys.executable, "predict_all.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                st.success("✅ Predictions generated successfully.")
                st.code(result.stdout, language="bash")
            except subprocess.CalledProcessError as e:
                st.error("❌ Prediction script failed.")
                st.code(e.stderr, language="bash")

    PRED_FILE = "data/predictions_today.csv"
    METRICS_FILE = "models/metrics.json"
    MODEL_FILE = "models/lstm_model.h5"

    if not os.path.exists(PRED_FILE) or os.path.getsize(PRED_FILE) == 0:
        st.warning("No predictions found. Please generate predictions first.")
    else:
        df = pd.read_csv(PRED_FILE)
        counts = df["Suggested Action"].value_counts()
        total = len(df)
        pie_labels = counts.index.tolist()
        pie_values = counts.values.tolist()

        st.markdown("### 📊 Model Signal Breakdown")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Tickers Covered", total)
            st.metric("Buy Count", counts.get("Buy", 0))
            st.metric("Sell Count", counts.get("Sell", 0))
            st.metric("Hold Count", counts.get("Hold", 0))

        with col2:
            fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=0.4)])
            fig.update_traces(textinfo="label+percent",
                              marker=dict(colors=["#90ee90", "#ffcccb", "#dddddd"]))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧩 Sector Confidence Map")
        sector_conf = df.groupby("Sector")["Confidence"].mean().sort_values()
        fig_bar = go.Figure(go.Bar(
            x=sector_conf.values,
            y=sector_conf.index,
            orientation='h',
            marker_color="#0e76fd"
        ))
        fig_bar.update_layout(
            title="Avg Model Confidence by Sector",
            xaxis_title="Confidence (%)",
            height=400,
            margin=dict(l=60, r=10, t=40, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 📈 Confidence Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["Confidence"],
            nbinsx=20,
            marker_color="#0e76fd"
        ))
        fig_hist.update_layout(xaxis_title="Model Confidence (%)", yaxis_title="Count", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### 🧠 Smart Summary")
        avg_conf = round(df["Confidence"].mean(), 2)
        top = df.sort_values("Confidence", ascending=False).head(3)
        summary = [
            f"Average model confidence today is **{avg_conf}%** across all signals.",
            *[
                f"- {row['Ticker']} ({row['Sector']}) → **{row['Suggested Action']}** • {row['Confidence']}% | Est. Return: {row['Expected Return %']}%"
                for _, row in top.iterrows()
            ]
        ]
        st.markdown("\n".join(summary))

        # ---------- Analyst Mode Panel ----------
        if view_mode == "Analyst":
            st.markdown("### ⚙️ Model Diagnostics")
            with st.expander("🔍 Health Report & Retraining", expanded=False):
                colA, colB = st.columns([2, 1])
                with colA:
                    if os.path.exists(METRICS_FILE):
                        import json
                        with open(METRICS_FILE) as f:
                            m = json.load(f)
                        st.write(f"**Last Retrained:** {m['timestamp']}")
                        st.write(f"**Rows Used:** {m['data_rows']}")
                        st.write(f"**Train Accuracy:** {round(m['train_acc']*100, 2)}%")
                        st.write(f"**Reg MAE:** {round(m['reg_mae'], 4)}")
                        st.write(f"**Volatility Accuracy:** {round(m['vol_acc']*100, 2)}%")
                    else:
                        st.info("No training metrics logged yet.")
                with colB:
                    st.markdown("⚠️ Model retraining is CPU-intensive and may pause the UI for several seconds.")
                    confirm = st.checkbox("Yes, retrain now", key="retrain_confirm")
                    if st.button("🔁 Retrain Model") and confirm:
                        with st.spinner("Retraining in progress..."):
                            subprocess.run([sys.executable, "model.py"])
                        st.success("Model retrained successfully.")

# ---------- Tab 1: Market View ----------
with tabs[1]:
    import os
    import pandas as pd

    PRED_FILE = "data/predictions_today.csv"
    st.subheader("📈 S&P 500 Signals Overview")

    if not os.path.exists(PRED_FILE) or os.path.getsize(PRED_FILE) == 0:
        st.warning("📭 No predictions found. Please run `predict_all.py` using the button in the Overview tab.")
    else:
        df = pd.read_csv(PRED_FILE)

        with st.expander("📋 Filters"):
            action = st.selectbox("Suggested Action", ["All", "Buy", "Sell", "Hold"])
            vol_filter = st.selectbox("Volatility Class", ["All", "Low", "Medium", "High"])
            min_conf = st.slider("Confidence Threshold (%)", 50, 100, 60)
            min_return = st.slider("Expected Return % (min)", -10, 10, 0)
            search = st.text_input("Search Ticker", key="search_tab5")

        filtered = df[(df["Confidence"] >= min_conf) & (df["Expected Return %"] >= min_return)]
        if action != "All":
            filtered = filtered[filtered["Suggested Action"] == action]
        if vol_filter != "All" and "Volatility Class" in filtered.columns:
            filtered = filtered[filtered["Volatility Class"] == vol_filter]
        if search:
            filtered = filtered[filtered["Ticker"].str.contains(search.upper())]

        top_buy = filtered[filtered["Suggested Action"] == "Buy"].sort_values("Confidence", ascending=False).head(1)
        top_sell = filtered[filtered["Suggested Action"] == "Sell"].sort_values("Confidence", ascending=False).head(1)
        top_hold = filtered[filtered["Suggested Action"] == "Hold"].sort_values("Confidence", ascending=False).head(1)

        st.markdown("### 🔍 Top Model Picks")
        cols = st.columns(3)

        def format_tile(df_row):
            return f"""
            <div style='
                border-radius:10px;
                background-color:#ffffff;
                padding:15px;
                border:1px solid #e0e0e0;
                box-shadow:0px 1px 3px rgba(0,0,0,0.05);
                height:110px;
            '>
                <strong style='font-size:20px'>{df_row["Ticker"]}</strong> • {df_row["Sector"]}<br/>
                <span style='font-size:16px'>
                {df_row["Prediction"]} <b>{df_row["Suggested Action"]}</b><br/>
                Confidence: <b>{df_row["Confidence"]}%</b><br/>
                Volatility: <b>{df_row["Volatility Class"]}</b>
                </span>
            </div>
            """

        if not top_buy.empty:
            cols[0].markdown(format_tile(top_buy.iloc[0]), unsafe_allow_html=True)
        if not top_sell.empty:
            cols[1].markdown(format_tile(top_sell.iloc[0]), unsafe_allow_html=True)
        if not top_hold.empty:
            cols[2].markdown(format_tile(top_hold.iloc[0]), unsafe_allow_html=True)

        # Styled DataFrame with colored volatility class
        def highlight_vol(val):
            if val == "Low":
                return "background-color:#d4edda"  # light green
            elif val == "Medium":
                return "background-color:#fff3cd"  # soft yellow
            elif val == "High":
                return "background-color:#f8d7da"  # light red
            return ""

        styled = filtered.style\
            .applymap(lambda x: "background-color:#90ee90" if x == "Buy"
                      else "background-color:#ffcccb" if x == "Sell" else "",
                      subset=["Suggested Action"])\
            .applymap(highlight_vol, subset=["Volatility Class"])

        st.dataframe(styled, use_container_width=True, height=600)

# ---------- Tab 2: Stock Analyzer ----------
with tabs[2]:
    st.subheader("🔍 Stock Analyzer")

    DATA_FILE = "data/sp500_data.csv"
    MODEL_FILE = "models/lstm_model.h5"
    PRED_LOG = "data/predictions_log.csv"

    if not os.path.exists(DATA_FILE):
        st.warning("Historical data not found. Run collector.py.")
    else:
        df = pd.read_csv(DATA_FILE).sort_values("Timestamp")
        tickers = df['Ticker'].unique()
        ticker = st.selectbox("Select stock", tickers, key="stock_analyzer_select")

        df_t = df[df['Ticker'] == ticker].sort_values("Timestamp").tail(100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_t["Timestamp"], y=df_t["Price"], mode="lines", name="Price"))
        fig.add_trace(go.Bar(x=df_t["Timestamp"], y=df_t["Sentiment"],
                             name="Sentiment", yaxis="y2", opacity=0.5))
        fig.update_layout(
            title=f"{ticker} Price & Sentiment",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Sentiment", overlaying="y", side="right", range=[-1, 1])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional confidence trend
        if os.path.exists(PRED_LOG):
            preds_df = pd.read_csv(PRED_LOG)
            pred_history = preds_df[preds_df['Ticker'] == ticker].sort_values("Timestamp").tail(30)
            if len(pred_history) >= 3:
                st.subheader("📉 Prediction Confidence Trend")
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    x=pred_history["Timestamp"],
                    y=pred_history["Confidence"],
                    mode="lines+markers",
                    line=dict(color="#0e76fd"),
                    name="Confidence %"
                ))
                fig_conf.update_layout(
                    yaxis=dict(title="Confidence %", range=[0, 100]),
                    xaxis=dict(title="Time"),
                    height=300,
                    margin=dict(t=10, b=40, l=50, r=10)
                )
                st.plotly_chart(fig_conf, use_container_width=True)

        # Predict next move
        if os.path.exists(MODEL_FILE):
            if st.button("🔮 Predict Next Move"):
                try:
                    model = load_model(MODEL_FILE)
                    last = df_t[['Price', 'Sentiment', 'MA', 'STD', 'RSI']].dropna().tail(10)
                    if len(last) < 10:
                        st.warning("Not enough recent data with all features.")
                    else:
                        scaler = MinMaxScaler()
                        scaled = scaler.fit_transform(last)
                        X = np.expand_dims(scaled, axis=0)
                        out = model.predict(X)
                        if isinstance(out, list):
                            pred_class = out[0]
                            pred_return = out[1]
                        else:
                            pred_class, pred_return = out, [[0]]
                        direction = "📈 Up" if pred_class[0][0] > 0.5 else "📉 Down"
                        confidence = round(pred_class[0][0] * 100, 2)
                        est_return = round(pred_return[0][0] * 100, 2)
                        st.success(f"Prediction: {direction} with {confidence}% confidence")
                        st.info(f"Estimated Price Move: {est_return}%")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.warning("Model not found. Run model.py to train it.")

# ---------- Tab 3: Portfolio Tracker ----------
with tabs[3]:
    st.subheader("💼 Portfolio Tracker")

    import json
    import plotly.express as px

    PORTFOLIO_FILE = "data/portfolio.json"
    df = pd.read_csv(DATA_FILE).sort_values("Timestamp")
    latest_prices = df.groupby("Ticker").tail(1).set_index("Ticker")

    # Load or initialize portfolio
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            portfolio = json.load(f)
    else:
        portfolio = {"cash": 100000, "positions": {}, "history": []}

    cash = st.number_input("💰 Starting Cash", value=portfolio.get("cash", 5000), step=1000)    
    ticker = st.selectbox("Choose ticker", df["Ticker"].unique(), key="portfolio_select")
    shares = st.number_input("Shares", step=10, value=0)

    col1, col2 = st.columns(2)

    if col1.button("🛒 Buy"):
        price = latest_prices.loc[ticker]["Price"]
        total = price * shares
        if total > portfolio["cash"]:
            st.error("🚫 Not enough cash.")
        else:
            portfolio["cash"] -= total
            pos = portfolio["positions"].get(ticker, {"shares": 0, "avg_price": 0})
            total_shares = pos["shares"] + shares
            avg_price = (pos["shares"] * pos["avg_price"] + shares * price) / total_shares
            portfolio["positions"][ticker] = {"shares": total_shares, "avg_price": avg_price}
            portfolio["history"].append({"action": "BUY", "ticker": ticker, "price": price, "shares": shares})
            st.success(f"Bought {shares} shares of {ticker} at ${price:.2f}")

    if col2.button("💼 Sell"):
        if ticker not in portfolio["positions"] or portfolio["positions"][ticker]["shares"] < shares:
            st.error("🚫 Not enough shares.")
        else:
            price = latest_prices.loc[ticker]["Price"]
            proceeds = price * shares
            portfolio["cash"] += proceeds
            portfolio["positions"][ticker]["shares"] -= shares
            if portfolio["positions"][ticker]["shares"] == 0:
                del portfolio["positions"][ticker]
            portfolio["history"].append({"action": "SELL", "ticker": ticker, "price": price, "shares": shares})
            st.success(f"Sold {shares} shares of {ticker} at ${price:.2f}")

    # Save updated state
    portfolio["cash"] = cash
    os.makedirs("data", exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

    # Summary
    st.markdown("### 📊 Portfolio Summary")
    rows = []
    total_value = 0
    if "positions" not in portfolio:
        portfolio["positions"] = {}
    for tkr, data in portfolio["positions"].items():
        price = latest_prices.loc[tkr]["Price"]
        val = price * data["shares"]
        pnl = (price - data["avg_price"]) * data["shares"]
        sector = latest_prices.loc[tkr].get("Sector", "Unknown")
        rows.append({
            "Ticker": tkr,
            "Sector": sector,
            "Shares": data["shares"],
            "Last Price": round(price, 2),
            "Avg Cost": round(data["avg_price"], 2),
            "Value": round(val, 2),
            "Unrealized PnL": round(pnl, 2)
        })
        total_value += val

    df_holdings = pd.DataFrame(rows)
    st.dataframe(df_holdings, use_container_width=True)
    colA, colB = st.columns(2)
    colA.metric("Cash", f"${round(portfolio['cash'], 2)}")
    colB.metric("Total Equity", f"${round(portfolio['cash'] + total_value, 2)}")

    if not df_holdings.empty:
        st.markdown("### 🧩 Allocation by Sector")
        fig = px.pie(df_holdings, values="Value", names="Sector", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📜 Trade History"):
        if portfolio.get("history"):
            st.dataframe(pd.DataFrame(portfolio["history"])[::-1], use_container_width=True)
        else:
            st.info("No trades recorded yet.")

# ---------- Tab 4: Alerts & Sector Breakdown ----------
with tabs[4]:
    st.subheader("🚨 Recent High-Confidence Alerts")
    alert_file = "data/alerts_log.csv"
    if os.path.exists(alert_file):
        alerts = pd.read_csv(alert_file).sort_values("Timestamp", ascending=False).head(100)
        st.dataframe(alerts, use_container_width=True, height=400)
    else:
        st.info("No alerts logged yet. Run alert.py.")

        st.subheader("🏢 Sector-wise Outlook")
    pred_file = "data/predictions_today.csv"
    if os.path.exists(pred_file):
        preds = pd.read_csv(pred_file)
        if "Sector" in preds.columns:
            sector_summary = preds.groupby(["Sector", "Suggested Action"]).size().unstack(fill_value=0)
            st.bar_chart(sector_summary)
        else:
            st.warning("No sector data available in predictions file.")

# ---------- Tab 5: Momentum & Volatility Scanner ----------
with tabs[5]:
    st.subheader("📊 Momentum & Volatility Scanner")

    DATA_FILE = "data/sp500_data.csv"
    if not os.path.exists(DATA_FILE):
        st.warning("Historical data not found. Run collector.py.")
    else:
        df = pd.read_csv(DATA_FILE).sort_values("Timestamp")
        lookback = st.slider("Lookback Days", min_value=3, max_value=30, value=5)
        sort_option = st.selectbox("Sort by", ["Price Δ (%)", "Volatility", "Sentiment Δ"])
        search = st.text_input("Search Ticker")

        movers = []
        all_vols = []

        for ticker in df['Ticker'].unique():
            sub = df[df['Ticker'] == ticker].tail(lookback + 1)
            if len(sub) < lookback + 1:
                continue
            price_change = (sub['Price'].iloc[-1] - sub['Price'].iloc[0]) / sub['Price'].iloc[0] * 100
            sentiment_change = sub['Sentiment'].iloc[-1] - sub['Sentiment'].iloc[0]
            volatility = sub['Price'].pct_change().std()
            all_vols.append(volatility)
            movers.append({
                "Ticker": ticker,
                "Price Δ (%)": round(price_change, 2),
                "Sentiment Δ": round(sentiment_change, 2),
                "Volatility": round(volatility, 4)
            })

        df_movers = pd.DataFrame(movers)

        if not df_movers.empty:
            q_low = np.quantile(all_vols, 0.33)
            q_high = np.quantile(all_vols, 0.66)

            def vol_class(v):
                if v <= q_low:
                    return "Low"
                elif v >= q_high:
                    return "High"
                return "Medium"

            df_movers["Volatility Class"] = df_movers["Volatility"].apply(vol_class)

            if search:
                df_movers = df_movers[df_movers["Ticker"].str.contains(search.upper())]

            df_movers = df_movers.sort_values(sort_option, ascending=False).head(30)

            def highlight_vol(val):
                if val == "Low":
                    return "background-color:#d4edda"
                elif val == "Medium":
                    return "background-color:#fff3cd"
                elif val == "High":
                    return "background-color:#f8d7da"
                return ""

            styled = df_movers.style.applymap(highlight_vol, subset=["Volatility Class"])
            st.dataframe(styled, use_container_width=True, height=550)
        else:
            st.info("No movers available for the selected lookback window.")

# ---------- Tab 6: Explainability ----------
with tabs[6]:
    st.subheader("🧠 Explain Prediction")

    import shap
    import matplotlib.pyplot as plt
    from keras.losses import MeanSquaredError

    LOG_FILE = "data/predictions_log.csv"
    PRED_FILE = "data/predictions_today.csv"

    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        st.warning("Historical data or model not found.")
    else:
        df = pd.read_csv(DATA_FILE).sort_values("Timestamp")
        tickers = df['Ticker'].unique()
        ticker = st.selectbox("Choose a stock to analyze", tickers, key="explain_ticker")

        features = ['Price', 'Sentiment', 'MA', 'STD', 'RSI']
        df_t = df[df['Ticker'] == ticker].dropna().sort_values("Timestamp").tail(10)
        available_features = [col for col in features if col in df_t.columns]

        if len(df_t) < 6:
            st.warning("Not enough data to explain prediction (need at least 6 rows).")
        elif not available_features:
            st.error("No valid features available to explain prediction.")
        else:
            if len(available_features) < len(features):
                st.warning(f"Missing features: {set(features) - set(available_features)}")

            model = load_model(MODEL_FILE, custom_objects={"loss": MeanSquaredError()}, compile=False)
            latest_features = df_t[available_features]
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(latest_features)

            # Build prediction input with dummy sector data
            model_input_shape = model.input_shape  # (None, 6, total_features)
            seq_len, total_features = model_input_shape[1], model_input_shape[2]

            if scaled.shape[0] < seq_len:
                st.warning(f"Need at least {seq_len} timesteps for prediction.")
                st.stop()

            X_seq = scaled[-seq_len:]
            num_dummy = total_features - X_seq.shape[1]
            if num_dummy < 0:
                st.error("Model expects fewer features than provided. Input mismatch.")
                st.stop()

            sector_dummy = np.zeros((seq_len, num_dummy))
            X = np.concatenate([X_seq, sector_dummy], axis=1).reshape(1, seq_len, total_features)

            pred_class, pred_return, *_ = model.predict(X, verbose=0)
            confidence = round(pred_class[0][0] * 100, 2)
            est_return = round(pred_return[0][0] * 100, 2)
            direction = "📈 Up" if pred_class[0][0] > 0.5 else "📉 Down"

            st.markdown(f"""
            ### 🧠 Model says: {direction}
            - Confidence: **{confidence}%**
            - Estimated Return: **{est_return}%**
            """)

            if view_mode == "Analyst":
                st.subheader("📉 Confidence Trend (Last 30 Days)")
                if os.path.exists(LOG_FILE):
                    pred_log = pd.read_csv(LOG_FILE)
                    pred_log = pred_log[pred_log["Ticker"] == ticker].sort_values("Timestamp").tail(30)
                    if not pred_log.empty:
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=pred_log["Timestamp"],
                            y=pred_log["Confidence"],
                            mode="lines+markers",
                            line=dict(color="blue")
                        ))
                        fig_trend.update_layout(
                            yaxis=dict(title="Confidence (%)", range=[0, 100]),
                            xaxis=dict(title="Date"),
                            height=300
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

                st.subheader("🔬 Feature Breakdown (Most Recent Timestep)")
                last_row = latest_features.iloc[-1]
                sorted_feat = sorted(last_row.to_dict().items(), key=lambda x: abs(x[1]), reverse=True)
                fig = go.Figure(go.Bar(
                    x=[v for _, v in sorted_feat],
                    y=[k for k, _ in sorted_feat],
                    orientation='h',
                    marker=dict(color="orange")
                ))
                fig.update_layout(yaxis=dict(autorange="reversed"), height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("💡 SHAP Attribution")
                try:
                    background = np.tile(X[0], (10, 1, 1))
                    explainer = shap.GradientExplainer(model, background)
                    shap_values = explainer.shap_values(X)
                    shap_class = shap_values[0][0]
                    mean_abs = np.abs(shap_class).mean(axis=0)

                    fig, ax = plt.subplots()
                    shap.bar_plot(mean_abs, feature_names=available_features, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP explanation failed: {e}")

                st.subheader("📡 Temporal Attention Saliency")
                try:
                    from tensorflow.keras.models import Model
                    attn_weights = model.get_layer("attention_weights").output
                    intermediate = Model(inputs=model.input, outputs=attn_weights)
                    attn_output = intermediate.predict(X, verbose=0)
                    weights = attn_output[0][-1]

                    fig_attn = go.Figure(go.Bar(
                        x=[f"T-{seq_len - 1 - i}" for i in range(len(weights))],
                        y=weights,
                        marker_color="#ff914d"
                    ))
                    fig_attn.update_layout(
                        xaxis_title="Timestep",
                        yaxis_title="Attention Weight",
                        height=300
                    )
                    st.plotly_chart(fig_attn, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not extract attention weights: {e}")

                if os.path.exists(PRED_FILE):
                    preds = pd.read_csv(PRED_FILE)
                    row = preds[preds["Ticker"] == ticker].head(1)
                    if not row.empty:
                        vol_class = row["Volatility Class"].values[0]
                        action = row["Suggested Action"].values[0]
                        st.info(f"📦 Volatility Class: **{vol_class}**\n\n🔁 Suggested Action: **{action}**")

# ---------- Tab 7: Backtest ----------
with tabs[6]:  # Make sure this is the correct index for "Backtest"
    st.subheader("📉 Model Backtest Results")

    SUMMARY_FILE = "data/backtest_summary.csv"
    TRADE_LOG_FILE = "data/backtest_trades.csv"
    EQUITY_DIR = "data/equity_curves"

    # Handle empty or missing summary file
    if os.path.exists(SUMMARY_FILE) and os.path.getsize(SUMMARY_FILE) > 0:
        try:
            df_summary = pd.read_csv(SUMMARY_FILE)
            if df_summary.empty:
                st.warning("Backtest summary is empty. Run backtest.py to generate results.")
            else:
                st.markdown("### 📊 Backtest Summary (Top 10)")
                st.dataframe(df_summary.head(10), use_container_width=True)

                st.markdown("### 🥇 Select a Ticker to View Equity Curve")
                selected = st.selectbox("Ticker", df_summary["Ticker"].tolist(), key="backtest_ticker")
                equity_path = f"{EQUITY_DIR}/{selected}_equity.png"
                if os.path.exists(equity_path):
                    st.image(equity_path, use_column_width=True)
                else:
                    st.info("No equity plot available for this ticker.")
        except Exception as e:
            st.error(f"Could not load backtest summary: {e}")
    else:
        st.warning("Backtest summary not found or is empty. Run backtest.py to generate results.")

    # Handle trade log
    if os.path.exists(TRADE_LOG_FILE) and os.path.getsize(TRADE_LOG_FILE) > 0:
        try:
            df_trades = pd.read_csv(TRADE_LOG_FILE)
            if not df_trades.empty:
                st.markdown("### 📜 Recent Trade Log")
                st.dataframe(df_trades.tail(30), use_container_width=True)
            else:
                st.info("Trade log is empty.")
        except Exception as e:
            st.error(f"Could not load trade log: {e}")
    else:
        st.info("No trade log found or it is empty. Run backtest.py to generate results.")
