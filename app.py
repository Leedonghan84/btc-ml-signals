import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Bitcoin Prediction", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("btc_technical_indicators.csv")

    # ✅ 디버깅 출력
    st.write("📊 데이터 컬럼 목록:", df.columns.tolist())
    st.write("🔍 데이터 상위 5개:", df.head())

    # ✅ 'Close' 컬럼 유무 확인 및 결측값 처리
    if 'Close' not in df.columns:
        st.error("❌ 'Close' 컬럼이 데이터에 없습니다.")
        st.stop()

    df = df.dropna(subset=['Close'])  # Close에 NaN 있는 행 제거

    try:
        df['Future_Return'] = df['Close'].shift(-3) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0).astype(int)
    except Exception as e:
        st.error(f"❌ 예측 열 생성 중 오류 발생: {e}")
        st.stop()

    return df

df = load_data()

st.title("📈 Bitcoin Technical Indicator Prediction")

st.write("### Raw Data")
st.dataframe(df.head())

# Features and target
features = df.drop(columns=["Future_Return", "Target"]).select_dtypes(include=[np.number])
target = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Model Evaluation")
st.text("Confusion Matrix:")
st.text(confusion_matrix(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = features.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)
