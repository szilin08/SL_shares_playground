import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px
from transformers import pipeline

# Cache the model to avoid reloading
@st.cache_resource
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_generator():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        # Set pad_token_id explicitly
        model.config.pad_token_id = tokenizer.eos_token_id
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.error(f"Error loading GPT-2 model: {e}")
        return None


# Fetch data function
def fetch_data(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)

# Generate AI explanation
def generate_gpt2_explanation(prompt, max_new_tokens=100):  # Reduced for memory
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
                f"Analyze stock performance for LBS Bina and competitors from {start} to {end}. "
                f"Percentage changes in closing prices: {', '.join([f'{comp}: {pct:.2f}%' for comp, pct in pct_changes.items()])}. "
                f"Top performer: {pct_changes.idxmax()} ({pct_changes.max():.2f}%). "
                f"Weakest performer: {pct_changes.idxmin()} ({pct_changes.min():.2f}%). "
                f"Explain these trends simply for investors."
            )
            st.write(generate_gpt2_explanation(prompt_close))

            st.markdown("### Volume Explanation")
            avg_volumes = df_all.groupby('Company')['Volume'].mean().sort_values(ascending=False)
            max_volumes = df_all.groupby('Company')['Volume'].max()
            prompt_volume = (
                f"Explain trading volume trends for LBS Bina and competitors. "
                f"Average daily volumes: {', '.join([f'{comp}: {vol:,.0f} shares' for comp, vol in avg_volumes.items()])}. "
                f"Highest average volume: {avg_volumes.idxmax()}. "
                f"Largest single-day volume: {max_volumes.idxmax()} ({max_volumes[max_volumes.idxmax]:,.0f} shares). "
                f"Describe what volume indicates about investor interest."
            )
            st.write(generate_gpt2_explanation(prompt_volume))

            st.markdown("### Stock Volatility (Annualized)")
            df_all['Daily_Return'] = df_all.groupby('Company')['Close'].pct_change()
            volatilities = (df_all.groupby('Company')['Daily_Return'].std() * (252 ** 0.5)).sort_values(ascending=False)
            prompt_volatility = (
                f"Explain stock volatility for LBS Bina and competitors. "
                f"Annualized volatilities: {', '.join([f'{comp}: {vol:.2%}' for comp, vol in volatilities.items()])}. "
                f"Highest volatility: {volatilities.idxmax()}. "
                f"Describe what volatility means for investors and risks."
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
                f"Analyze 50-day moving average trends for LBS Bina and competitors. "
                f"Percentage of days above 50-day MA: {', '.join([f'{comp}: {pct:.2f}%' for comp, pct in ma_trends.sort_values(ascending=False).items()])}. "
                f"Strongest trend: {ma_trends.idxmax()}. Weakest: {ma_trends.idxmin()}. "
                f"Explain what this indicates about stock momentum."
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
                f"Explain maximum drawdown for LBS Bina and competitors. "
                f"Drawdowns: {', '.join([f'{comp}: {drawdown:.2f}%' for comp, drawdown in max_drawdowns.items()])}. "
                f"Largest: {max_drawdowns.idxmax()}. Smallest: {max_drawdowns.idxmin()}. "
                f"Describe what drawdown means for investors and risks."
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
                f"Explain average daily returns for LBS Bina and competitors. "
                f"Returns: {', '.join([f'{comp}: {ret:.4f}%' for comp, ret in avg_daily_returns.items()])}. "
                f"Highest: {avg_daily_returns.idxmax()}. Lowest: {avg_daily_returns.idxmin()}. "
                f"Explain what daily returns indicate about performance."
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

