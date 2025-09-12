import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px
from transformers import pipeline

# Cache the model to avoid reloading
@st.cache_resource
def load_generator():
    try:
        return pipeline('text-generation', model='distilgpt2', device=-1)  # Force CPU
    except Exception as e:
        st.error(f"Error loading GPT-2 model: {e}")
        return None

# Fetch data function
def fetch_data(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)

# Generate AI explanation
def generate_gpt2_explanation(prompt, max_new_tokens=150):  # Reduced for memory efficiency
    generator = load_generator()
    if generator is None:
        return "Unable to generate explanation due to model loading error."
    try:
        result = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, truncation=True)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating explanation: {e}"

# Main app
def main():
    st.set_page_config(layout="wide")
    st.title("📈 Competitor Stock Monitoring – LBS Bina as Base")

    base_ticker = "5789.KL"
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

    start = st.date_input("Start date", value=date(2020, 1, 1))
    end = st.date_input("End date", value=date.today())
    selected_competitors = st.multiselect(
        "Select competitors to compare against LBS Bina",
        list(competitors.keys())
    )

    if st.button("Get Historical Data"):
        try:
            df_base = fetch_data(base_ticker, start, end + pd.Timedelta(days=1))
            df_base["Company"] = "LBS Bina"
            dfs = [df_base]
            for comp in selected_competitors:
                ticker = competitors[comp]
                df = fetch_data(ticker, start, end + pd.Timedelta(days=1))
                df["Company"] = comp
                dfs.append(df)
            df_all = pd.concat(dfs, keys=[d["Company"].iloc[0] for d in dfs]).reset_index()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Closing Price Comparison")
                fig_close = px.line(
                    df_all, x="Date", y="Close", color="Company",
                    title="Closing Price", width=800, height=500
                )
                st.plotly_chart(fig_close, use_container_width=True)
            with col2:
                st.subheader("Volume Comparison")
                fig_vol = px.line(
                    df_all, x="Date", y="Volume", color="Company",
                    title="Trading Volume", width=800, height=500
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            st.subheader("📊 Historical Data Table")
            st.dataframe(df_all)
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Combined CSV", csv, file_name="competitor_comparison.csv", mime="text/csv"
            )

            st.subheader("🤖 GPT-2 Automated Analysis & Explanations")
            st.markdown("### Closing Price Explanation")
            first_closes = df_all.groupby('Company')['Close'].first()
            last_closes = df_all.groupby('Company')['Close'].last()
            pct_changes = ((last_closes - first_closes) / first_closes * 100).sort_values(ascending=False)
            prompt_close = (
                f"Analyze the stock performance of companies including LBS Bina and its competitors. "
                f"The percentage changes in closing prices from {start} to {end} are: "
                f"{', '.join([f'{comp}: {pct:.2f}%' for comp, pct in pct_changes.items()])}. "
                f"The top performer is {pct_changes.idxmax()} with {pct_changes.max():.2f}% gain, "
                f"and the weakest is {pct_changes.idxmin()} with {pct_changes.min():.2f}% change. "
                f"Explain these trends in simple terms, focusing on what the percentage changes mean for investors."
            )
            st.write(generate_gpt2_explanation(prompt_close))

            st.markdown("### Volume Explanation")
            avg_volumes = df_all.groupby('Company')['Volume'].mean().sort_values(ascending=False)
            max_volumes = df_all.groupby('Company')['Volume'].max()
            prompt_volume = (
                f"Explain the trading volume trends for stocks including LBS Bina and its competitors. "
                f"Average daily volumes are: {', '.join([f'{comp}: {vol:,.0f} shares' for comp, vol in avg_volumes.items()])}. "
                f"The highest average volume is {avg_volumes.idxmax()} and the largest single-day volume spike was for {max_volumes.idxmax()} at {max_volumes[max_volumes.idxmax]:,.0f} shares. "
                f"Describe what trading volume indicates about investor interest and market activity."
            )
            st.write(generate_gpt2_explanation(prompt_volume))

            st.markdown("### Stock Volatility (Annualized)")
            df_all['Daily_Return'] = df_all.groupby('Company')['Close'].pct_change()
            volatilities = (df_all.groupby('Company')['Daily_Return'].std() * (252 ** 0.5)).sort_values(ascending=False)
            prompt_volatility = (
                f"Explain stock volatility for companies including LBS Bina and its competitors. "
                f"Annualized volatilities are: {', '.join([f'{comp}: {vol:.2%}' for comp, vol in volatilities.items()])}. "
                f"The highest volatility is {volatilities.idxmax()}. Describe what volatility means for investors and the risks involved."
            )
            st.write(generate_gpt2_explanation(prompt_volatility))

            volatility_df = pd.DataFrame({'Company': volatilities.index, 'Volatility': volatilities.values})
            fig_volatility = px.bar(
                volatility_df, x='Company', y='Volatility', title="Annualized Volatility Comparison",
                width=800, height=500, color='Company', text_auto='.2%'
            )
            fig_volatility.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_volatility, use_container_width=True)

            st.markdown("### Moving Average Trends (50-Day)")
            df_all['MA50'] = df_all.groupby('Company')['Close'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop=True)
            df_all['Above_MA50'] = df_all['Close'] > df_all['MA50']
            ma_trends = df_all.groupby('Company')['Above_MA50'].mean() * 100
            prompt_ma = (
                f"Analyze the 50-day moving average trends for stocks including LBS Bina and its competitors. "
                f"The percentage of days above the 50-day moving average are: {', '.join([f'{comp}: {pct:.2f}%' for comp, pct in ma_trends.sort_values(ascending=False).items()])}. "
                f"The strongest trend is {ma_trends.idxmax()} and the weakest is {ma_trends.idxmin()}. "
                f"Explain what the moving average trend indicates about stock momentum."
            )
            st.write(generate_gpt2_explanation(prompt_ma))

            ma_trends_df = pd.DataFrame({'Company': ma_trends.index, 'Percentage': ma_trends.values})
            fig_ma_trends = px.bar(
                ma_trends_df, x='Company', y='Percentage', title="Percentage of Days Above 50-Day Moving Average",
                width=800, height=500, color='Company', text_auto='.2f'
            )
            fig_ma_trends.update_layout(yaxis_title="Percentage of Days (%)")
            st.plotly_chart(fig_ma_trends, use_container_width=True)

            st.markdown("### Maximum Drawdown")
            df_pivot = df_all.pivot(index='Date', columns='Company', values='Close')
            rolling_max = df_pivot.cummax()
            drawdowns = (df_pivot - rolling_max) / rolling_max
            max_drawdowns = (-drawdowns.min() * 100).sort_values(ascending=False)
            prompt_drawdown = (
                f"Explain the maximum drawdown for stocks including LBS Bina and its competitors. "
                f"Maximum drawdowns are: {', '.join([f'{comp}: {drawdown:.2f}%' for comp, drawdown in max_drawdowns.items()])}. "
                f"The largest drawdown is {max_drawdowns.idxmax()} and the smallest is {max_drawdowns.idxmin()}. "
                f"Describe what maximum drawdown means for investors and the risks it highlights."
            )
            st.write(generate_gpt2_explanation(prompt_drawdown))

            drawdown_df = pd.DataFrame({'Company': max_drawdowns.index, 'Drawdown': max_drawdowns.values})
            fig_drawdown = px.bar(
                drawdown_df, x='Company', y='Drawdown', title="Maximum Drawdown Comparison",
                width=800, height=500, color='Company', text_auto='.2f'
            )
            fig_drawdown.update_layout(yaxis_title="Maximum Drawdown (%)")
            st.plotly_chart(fig_drawdown, use_container_width=True)

            st.markdown("### Average Daily Returns")
            avg_daily_returns = (df_all.groupby('Company')['Daily_Return'].mean() * 100).sort_values(ascending=False)
            prompt_returns = (
                f"Explain the average daily returns for stocks including LBS Bina and its competitors. "
                f"Average daily returns are: {', '.join([f'{comp}: {ret:.4f}%' for comp, ret in avg_daily_returns.items()])}. "
                f"The highest is {avg_daily_returns.idxmax()} and the lowest is {avg_daily_returns.idxmin()}. "
                f"Explain what average daily returns indicate about stock performance."
            )
            st.write(generate_gpt2_explanation(prompt_returns))

            returns_df = pd.DataFrame({'Company': avg_daily_returns.index, 'Average Daily Return': avg_daily_returns.values})
            fig_returns = px.bar(
                returns_df, x='Company', y='Average Daily Return', title="Average Daily Returns Comparison",
                width=800, height=500, color='Company', text_auto='.4f'
            )
            fig_returns.update_layout(yaxis_title="Average Daily Return (%)")
            st.plotly_chart(fig_returns, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()
