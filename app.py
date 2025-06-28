import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="BTC 예측 앱", layout="wide")
st.title("📈 Bitcoin Technical Indicator Viewer + 예측")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("btc_technical_indicators.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    df['Future_Return'] = df['Close'].shift(-3) / df['Close'] - 1  # 3일 후 수익률
    df.dropna(inplace=True)
    return df

df = load_data()

# --- 시각화 ---
st.subheader("1️⃣ 기술적 지표 시각화")
indicators = ['Close', 'EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD', 'Signal', 'BB_Width', 'Volume']
selected = st.multiselect("📊 표시할 기술적 지표 선택", indicators, default=['Close', 'EMA20', 'EMA50'])

fig, ax = plt.subplots(figsize=(14, 6))
df[selected].plot(ax=ax)
st.pyplot(fig)

st.subheader("2️⃣ 예측 모델 성능 평가")
feature_cols = ['EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD', 'Signal', 'BB_Width', 'Volume']
target_col = 'Future_Return'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

st.write(f"✅ R²: `{r2:.3f}` | RMSE: `{rmse:.4f}` | MAE: `{mae:.4f}`")

# --- 예측 vs 실제 시각화 ---
st.subheader("3️⃣ 예측 vs 실제 수익률")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_test.index, y_test.values, label='Actual', marker='o')
ax2.plot(y_test.index, y_pred, label='Predicted', marker='x')
ax2.set_title("3일 후 수익률 예측")
ax2.legend()
st.pyplot(fig2)

# --- 변수 중요도 ---
st.subheader("4️⃣ 변수 중요도")
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
importance_df.sort_values("Importance", ascending=False, inplace=True)
fig3, ax3 = plt.subplots()
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax3)
st.pyplot(fig3)

# --- 사용자 입력 예측 ---
st.subheader("5️⃣ 조건 입력 → 3일 후 수익률 예측")
user_input = {col: st.number_input(col, value=float(df[col].mean())) for col in feature_cols}
input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]

st.success(f"📊 예측 수익률 (3일 후): `{prediction*100:.2f}%`")
