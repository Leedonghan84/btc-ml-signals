# 필요한 패키지 설치 (처음 한 번만)
# pip install yfinance pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np

# 1. BTC 가격 수집
btc = yf.download('BTC-USD', start='2020-01-01', end='2025-01-01')

# 2. EMA 계산
btc['EMA20'] = btc['Close'].ewm(span=20, adjust=False).mean()
btc['EMA50'] = btc['Close'].ewm(span=50, adjust=False).mean()
btc['EMA200'] = btc['Close'].ewm(span=200, adjust=False).mean()

# 3. RSI(14)
delta = btc['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
btc['RSI14'] = 100 - (100 / (1 + rs))

# 4. MACD + Signal
ema12 = btc['Close'].ewm(span=12, adjust=False).mean()
ema26 = btc['Close'].ewm(span=26, adjust=False).mean()
btc['MACD'] = ema12 - ema26
btc['Signal'] = btc['MACD'].ewm(span=9, adjust=False).mean()

# 5. Bollinger Band Width
rolling_mean = btc['Close'].rolling(window=20).mean()
rolling_std = btc['Close'].rolling(window=20).std()
upper_band = rolling_mean + (rolling_std * 2)
lower_band = rolling_mean - (rolling_std * 2)
btc['BB_Width'] = (upper_band - lower_band) / rolling_mean

# 6. 정리 및 저장
btc_features = btc[['Close', 'Volume', 'EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD', 'Signal', 'BB_Width']]
btc_features.dropna(inplace=True)

# 7. CSV 저장
btc_features.to_csv("btc_technical_indicators.csv")
print("✅ 기술적 지표 계산 완료. 'btc_technical_indicators.csv'로 저장되었습니다.")
