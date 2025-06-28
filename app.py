import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BTC 기술적 분석", layout="wide")
st.title("📈 Bitcoin Technical Indicator Viewer")

# 데이터 로드
@st.cache_data
def load_data():
    return pd.read_csv("btc_technical_indicators.csv", index_col=0, parse_dates=True)

df = load_data()

# 지표 선택
indicators = ['Close', 'EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD', 'Signal', 'BB_Width', 'Volume']
selected = st.multiselect("📊 표시할 기술적 지표 선택", indicators, default=['Close', 'EMA20', 'EMA50'])

# 차트 표시
st.subheader("📉 기술적 지표 차트")
fig, ax = plt.subplots(figsize=(14, 6))
df[selected].plot(ax=ax)
st.pyplot(fig)

# 데이터 테이블
st.subheader("🔍 Raw Data Table")
st.dataframe(df.tail(100))
