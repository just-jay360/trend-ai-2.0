from ai_assistant import assistant
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import ta


class LivePredictor:

    def __init__(self):
        self.model = joblib.load("models/rf_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.feature_cols = [
            'SMA_20',
            'SMA_50',
            'EMA_20',
            'RSI',
            'MACD',
            'BB_high',
            'BB_low',
            'BB_width',
            'ATR',
            'Stochastic',
            'High_Low_Ratio',
            'Close_Open_Ratio',
            'Return_1',
            'Return_5'
        ]

    def get_latest_data(self):

        df = yf.download(
            "EURUSD=X",
            period="60d",
            interval="1h",
            progress=False,
            auto_adjust=True
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return df

    def engineer_features(self, df):

        df = df.copy()

        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], 20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], 50)

        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], 20)

        df['RSI'] = ta.momentum.rsi(df['Close'], 14)

        df['MACD'] = ta.trend.macd_diff(df['Close'])

        df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])

        df['BB_width'] = df['BB_high'] - df['BB_low']

        df['ATR'] = ta.volatility.average_true_range(
            df['High'],
            df['Low'],
            df['Close']
        )

        df['Stochastic'] = ta.momentum.stoch(
            df['High'],
            df['Low'],
            df['Close']
        )

        df['High_Low_Ratio'] = (
            df['High'] - df['Low']
        ) / df['Close']

        df['Close_Open_Ratio'] = (
            df['Close'] - df['Open']
        ) / df['Open']

        df['Return_1'] = df['Close'].pct_change()

        df['Return_5'] = df['Close'].pct_change(5)

        df.dropna(inplace=True)

        return df

    def predict(self):

        df = self.get_latest_data()

        df = self.engineer_features(df)

        latest = df.iloc[-1]

        X = latest[self.feature_cols].values.reshape(1, -1)

        X = self.scaler.transform(X)

        prediction = self.model.predict(X)[0]

        probability = self.model.predict_proba(X)[0]

        confidence = float(np.max(probability))

        if prediction == 1:
            signal = "BUY"
        else:
            signal = "SELL"

        if confidence < 0.60:
            signal = "HOLD"

        confidence_percent = round(confidence * 100, 2)

        analysis = assistant.explain(
            signal,
            confidence_percent,
            latest
        )

        return {
            "signal": signal,
            "confidence": confidence_percent,
            "price": round(float(latest["Close"]), 5),
            "time": str(df.index[-1]),
            "analysis": analysis
        }


live_predictor = LivePredictor()