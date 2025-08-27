import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px
import requests
import json
import os
import time
from threading import Lock

# --- Lock for sequential LLM calls ---
llm_lock = Lock()

# --- Check if running on Streamlit Cloud ---
def is_streamlit_cloud():
    return os.getenv('STREAMLIT_CLOUD') is not None or 'streamlit' in os.getenv('SERVER_SOFTWARE', '').lower()

# --- Check Ollama Server Readiness (local only) ---
def check_ollama_server(ollama_host):
    if is_streamlit_cloud():
        return False
    try:
        response = requests.get(f"http://{ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except:
        return False

# --- Warm Up Ollama Server (local only) ---
def warmup_ollama_server(ollama_host):
    if is_streamlit_cloud():
        return
    try:
        response = requests.post(
            f"http://{ollama_host}/api/generate",
            json={"model": "gemma:2b", "prompt": "Warmup test", "stream": False},
            timeout=10
        )
        response.raise_for_status()
    except:
        pass

# --- Get LLM Explanation (local only) ---
def get_llm_explanation(metric_name, data_dict):
    if is_streamlit_cloud():
        st.info("LLM explanations are unavailable on Streamlit Cloud. Using rule-based explanations.")
        return None
    ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')
    ollama_url = f"http://{ollama_host}/api/generate"
    prompt = f"""
    Explain '{metric_name}' in 80-120 words.
    Data: {json.dumps(data_dict, indent=2)}
    - Define the metric.
    - Note top/bottom performers.
    - Compare LBS Bina to the average.
    Keep it clear.
    """
    max_retries = 3
    with llm_lock:
        if not check_ollama_server(ollama_host):
            st.warning(f"Ollama server not ready on {ollama_host}. Warming up and retrying...")
            warmup_ollama_server(ollama_host)
            time.sleep(3)
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    ollama_url,
                    json={
                        "model": "gemma:2b",  # Use 'llama3' if system supports it
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=90
                )
                response.raise_for_status()
                return response.json().get("response", "Error generating explanation")
            except requests.exceptions.ReadTimeout:
                if attempt < max_retries - 1:
                    st.warning(f"Ollama timed out on {ollama_host}. Retrying in 3 seconds...")
                    time.sleep(3)
                    continue
                st.warning(f"Ollama timed out on {ollama_host} after {max_retries} attempts. Try closing apps or using 'gemma:2b' (`ollama pull gemma:2b`). Falling back to default explanation.")
                return None
            except requests.exceptions.ConnectionError:
                st.warning(f"Ollama server not running on {ollama_host}. Start it via the Ollama app or `ollama serve`. Ensure 'gemma:2b' is installed (`ollama pull gemma:2b`). Falling back to default explanation.")
                return None
            except requests.exceptions.HTTPError as e:
                st.warning(f"Ollama error: {str(e)}. Check if 'gemma:2b' is installed (`ollama list`) or run `ollama pull gemma:2b`. Falling back to default explanation.")
                return None
            except Exception as e:
                st.warning(f"Unexpected error with Ollama: {str(e)}. Falling back to default explanation.")
                return None

# --- Fetch Data Function ---
def fetch_data(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)

# --- Main App ---
def main():
    st.set_page_config(layout="wide")
    st.title("📈 Competitor Stock Monitoring – LBS Bina as Base")

    # Base company (LBS Bina)
    base_ticker = "5789.KL"

    # Competitors list
    competitors = {
        "S P Setia": "8664.KL",
        "Sime Darby Property": "5288.KL",
        "Eco World": "8206.KL",
        "UEM Sunrise": "5148.KL",
        "IOI Properties": "5249.KL",
        "Mah Sing": "8583.KL",
        "IJM Corporation": "3336.KL",
        "Sunway": "5211.KL",
        "Gamuda": "5398.KL",
        "OSK Holdings": "5053.KL",
        "UOA Development": "5200.KL",
    }

    # Date inputs
    start = st.date_input("Start date", value=date(2020, 1, 1))
    end = st.date_input("End date", value=date.today())

    # Select competitors
    selected_competitors = st.multiselect(
        "Select competitors to compare against LBS Bina",
        list(competitors.keys())
    )

    if st.button("Get Historical Data"):
        try:
            # Warm up Ollama server (local only)
            if not is_streamlit_cloud():
                warmup_ollama_server(os.getenv('OLLAMA_HOST', '127.0.0.1:11434'))

            # Fetch base company data
            df_base = fetch_data(base_ticker, start, end + pd.Timedelta(days=1))
            df_base["Company"] = "LBS Bina"

            # Store all dataframes
            dfs = [df_base]

            # Fetch competitors
            for comp in selected_competitors:
                ticker = competitors[comp]
                df = fetch_data(ticker, start, end + pd.Timedelta(days=1))
                df["Company"] = comp
                dfs.append(df)

            # Combine into single dataframe
            df_all = pd.concat(dfs, keys=[d["Company"].iloc[0] for d in dfs]).reset_index()

            # --- Side by Side Charts ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Closing Price Comparison")
                fig_close = px.line(
                    df_all,
                    x="Date", y="Close", color="Company",
                    title="Closing Price",
                    width=800, height=500
                )
                st.plotly_chart(fig_close, use_container_width=True)

            with col2:
                st.subheader("Volume Comparison")
                fig_vol = px.line(
                    df_all,
                    x="Date", y="Volume", color="Company",
                    title="Trading Volume",
                    width=800, height=500
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            # --- Historical Data at Bottom ---
            st.subheader("📊 Historical Data Table")
            st.dataframe(df_all)

            # Download CSV
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Combined CSV",
                csv,
                file_name="competitor_comparison.csv",
                mime="text/csv"
            )

            # --- Automated Analysis & Explanations ---
            st.subheader("🤖 Automated Analysis & Explanations")

            # Closing Price Insights
            st.markdown("### Closing Price Explanation")
            first_closes = df_all.groupby('Company')['Close'].first()
            last_closes = df_all.groupby('Company')['Close'].last()
            pct_changes = ((last_closes - first_closes) / first_closes * 100).sort_values(ascending=False)
            pct_changes_dict = pct_changes.to_dict()
            explanation_closing = get_llm_explanation("Closing Price Change (%)", pct_changes_dict)
            if explanation_closing:
                st.write(explanation_closing)
            else:
                st.write("This section analyzes the percentage change in closing prices for LBS Bina and competitors over the selected period:")
                for company, pct in pct_changes.items():
                    direction = "gain" if pct > 0 else "loss"
                    st.write(f"- **{company}**: {pct:.2f}% {direction}.")
                top_gainer = pct_changes.idxmax()
                top_gain = pct_changes.max()
                bottom_performer = pct_changes.idxmin()
                bottom_performance = pct_changes.min()
                st.write(f"**{top_gainer}** led with a {top_gain:.2f}% gain, indicating strong market confidence.")
                st.write(f"**{bottom_performer}** had a {bottom_performance:.2f}% change, suggesting challenges.")
                if 'LBS Bina' in pct_changes:
                    base_pct = pct_changes['LBS Bina']
                    avg_pct = pct_changes.mean()
                    performance_note = "outperformed" if base_pct > avg_pct else "underperformed" if base_pct < avg_pct else "matched"
                    st.write(f"LBS Bina ({base_pct:.2f}%) {performance_note} the average ({avg_pct:.2f}%).")

            # Volume Insights
            st.markdown("### Volume Explanation")
            avg_volumes = df_all.groupby('Company')['Volume'].mean().sort_values(ascending=False)
            max_volumes = df_all.groupby('Company')['Volume'].max()
            volumes_dict = {"Average Volumes": avg_volumes.to_dict(), "Max Volumes": max_volumes.to_dict()}
            explanation_volume = get_llm_explanation("Trading Volume", volumes_dict)
            if explanation_volume:
                st.write(explanation_volume)
            else:
                st.write("This section examines trading volumes over time:")
                for company, vol in avg_volumes.items():
                    st.write(f"- **{company}**: {vol:,.0f} shares/day (avg).")
                highest_avg_vol = avg_volumes.idxmax()
                highest_max_vol = max_volumes.idxmax()
                st.write(f"**{highest_avg_vol}** had the highest average volume, indicating high liquidity.")
                st.write(f"**{highest_max_vol}** had a peak volume of {max_volumes[highest_max_vol]:,.0f} shares.")

            # Volatility Insights
            st.markdown("### Stock Volatility (Annualized)")
            df_all['Daily_Return'] = df_all.groupby('Company')['Close'].pct_change()
            volatilities = (df_all.groupby('Company')['Daily_Return'].std() * (252 ** 0.5)).sort_values(ascending=False)
            volatilities_dict = volatilities.to_dict()
            explanation_volatility = get_llm_explanation("Annualized Volatility", volatilities_dict)
            if explanation_volatility:
                st.write(explanation_volatility)
            else:
                st.write("Volatility measures price fluctuations (risk):")
                for company, vol in volatilities.items():
                    st.write(f"- **{company}**: {vol:.2%} volatility.")
                highest_vol = volatilities.idxmax()
                st.write(f"**{highest_vol}** had the highest volatility, indicating higher risk.")
            
            # Volatility Chart
            volatility_df = pd.DataFrame({
                'Company': volatilities.index,
                'Volatility': volatilities.values
            })
            fig_volatility = px.bar(
                volatility_df,
                x='Company',
                y='Volatility',
                title="Annualized Volatility Comparison",
                width=800,
                height=500,
                color='Company',
                text_auto='.2%'
            )
            fig_volatility.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_volatility, use_container_width=True)

            # Moving Average Trends
            st.markdown("### Moving Average Trends (50-Day)")
            df_all['MA50'] = df_all.groupby('Company')['Close'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop=True)
            df_all['Above_MA50'] = df_all['Close'] > df_all['MA50']
            ma_trends = df_all.groupby('Company')['Above_MA50'].mean() * 100
            ma_trends_dict = ma_trends.to_dict()
            explanation_ma = get_llm_explanation("Days Above 50-Day MA (%)", ma_trends_dict)
            if explanation_ma:
                st.write(explanation_ma)
            else:
                st.write("This section shows the percentage of days above the 50-day moving average:")
                for company, pct in ma_trends.sort_values(ascending=False).items():
                    st.write(f"- **{company}**: {pct:.2f}% of days.")
                strongest_trend = ma_trends.idxmax()
                weakest_trend = ma_trends.idxmin()
                st.write(f"**{strongest_trend}** had the strongest bullish trend.")
                st.write(f"**{weakest_trend}** had the weakest trend.")
                if 'LBS Bina' in ma_trends:
                    base_ma = ma_trends['LBS Bina']
                    avg_ma = ma_trends.mean()
                    trend_note = "stronger" if base_ma > avg_ma else "weaker" if base_ma < avg_ma else "similar"
                    st.write(f"LBS Bina ({base_ma:.2f}%) had a {trend_note} trend vs. average ({avg_ma:.2f}%).")
            
            # Moving Average Trends Chart
            ma_trends_df = pd.DataFrame({
                'Company': ma_trends.index,
                'Percentage': ma_trends.values
            })
            fig_ma_trends = px.bar(
                ma_trends_df,
                x='Company',
                y='Percentage',
                title="Percentage of Days Above 50-Day Moving Average",
                width=800,
                height=500,
                color='Company',
                text_auto='.2f'
            )
            fig_ma_trends.update_layout(yaxis_title="Percentage of Days (%)")
            st.plotly_chart(fig_ma_trends, use_container_width=True)

            # Maximum Drawdown
            st.markdown("### Maximum Drawdown")
            df_pivot = df_all.pivot(index='Date', columns='Company', values='Close')
            rolling_max = df_pivot.cummax()
            drawdowns = (df_pivot - rolling_max) / rolling_max
            max_drawdowns = (-drawdowns.min() * 100).sort_values(ascending=False)
            max_drawdowns_dict = max_drawdowns.to_dict()
            explanation_drawdown = get_llm_explanation("Maximum Drawdown (%)", max_drawdowns_dict)
            if explanation_drawdown:
                st.write(explanation_drawdown)
            else:
                st.write("Maximum drawdown measures the largest price drop from a peak:")
                for company, drawdown in max_drawdowns.items():
                    st.write(f"- **{company}**: {drawdown:.2f}% max loss.")
                highest_drawdown = max_drawdowns.idxmax()
                lowest_drawdown = max_drawdowns.idxmin()
                st.write(f"**{highest_drawdown}** had the largest drawdown (highest risk).")
                st.write(f"**{lowest_drawdown}** had the smallest drawdown (more stable).")
                if 'LBS Bina' in max_drawdowns:
                    base_drawdown = max_drawdowns['LBS Bina']
                    avg_drawdown = max_drawdowns.mean()
                    risk_note = "riskier" if base_drawdown > avg_drawdown else "safer" if base_drawdown < avg_drawdown else "similar"
                    st.write(f"LBS Bina ({base_drawdown:.2f}%) was {risk_note} vs. average ({avg_drawdown:.2f}%).")
            
            # Maximum Drawdown Chart
            drawdown_df = pd.DataFrame({
                'Company': max_drawdowns.index,
                'Drawdown': max_drawdowns.values
            })
            fig_drawdown = px.bar(
                drawdown_df,
                x='Company',
                y='Drawdown',
                title="Maximum Drawdown Comparison",
                width=800,
                height=500,
                color='Company',
                text_auto='.2f'
            )
            fig_drawdown.update_layout(yaxis_title="Maximum Drawdown (%)")
            st.plotly_chart(fig_drawdown, use_container_width=True)

            # Average Daily Returns
            st.markdown("### Average Daily Returns")
            avg_daily_returns = (df_all.groupby('Company')['Daily_Return'].mean() * 100).sort_values(ascending=False)
            avg_daily_returns_dict = avg_daily_returns.to_dict()
            explanation_returns = get_llm_explanation("Average Daily Returns (%)", avg_daily_returns_dict)
            if explanation_returns:
                st.write(explanation_returns)
            else:
                st.write("Average daily returns show typical daily performance:")
                for company, ret in avg_daily_returns.items():
                    direction = "gain" if ret > 0 else "loss"
                    st.write(f"- **{company}**: {ret:.4f}% {direction}/day.")
                highest_daily = avg_daily_returns.idxmax()
                lowest_daily = avg_daily_returns.idxmin()
                st.write(f"**{highest_daily}** had the highest daily returns.")
                st.write(f"**{lowest_daily}** had the lowest daily returns.")
                if 'LBS Bina' in avg_daily_returns:
                    base_daily = avg_daily_returns['LBS Bina']
                    avg_daily = avg_daily_returns.mean()
                    daily_note = "stronger" if base_daily > avg_daily else "weaker" if base_daily < avg_daily else "similar"
                    st.write(f"LBS Bina ({base_daily:.4f}%) was {daily_note} vs. average ({avg_daily:.4f}%).")
            
            # Average Daily Returns Chart
            returns_df = pd.DataFrame({
                'Company': avg_daily_returns.index,
                'Average Daily Return': avg_daily_returns.values
            })
            fig_returns = px.bar(
                returns_df,
                x='Company',
                y='Average Daily Return',
                title="Average Daily Returns Comparison",
                width=800,
                height=500,
                color='Company',
                text_auto='.4f'
            )
            fig_returns.update_layout(yaxis_title="Average Daily Return (%)")
            st.plotly_chart(fig_returns, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()
