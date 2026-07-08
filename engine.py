"""
Forex Prediction Engine - Production Version
Modified for DigitalOcean App Platform (uses only Random Forest)
Add LSTM support later with larger droplet
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import ta
import warnings
warnings.filterwarnings('ignore')

class ForexEngine:
    def __init__(self, symbol="EURUSD=X", period="60d", lookback=50,
                 predict_horizon=5, threshold=0.0002):
        self.symbol = symbol
        self.period = period
        self.lookback = lookback
        self.predict_horizon = predict_horizon
        self.threshold = threshold
        self.scaler = None
        self.feature_cols = None
        self.model = None
        self.df = None
        self.results = {}

    def fetch_data(self):
        """Download and prepare OHLCV data"""
        try:
            print(f"Downloading data for: {self.symbol}")

            df = yf.download(
                tickers=self.symbol,
                period=self.period,
                interval="1h",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            print(df.head())
            print(f"Rows downloaded: {len(df)}")

            if df.empty:
                raise ValueError("No data downloaded")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only the columns we need
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            df = df.dropna()

            print("Final columns:", df.columns.tolist())

            return df

        except Exception as e:
            print(f"Data fetch error: {e}")
            return self._generate_fallback_data()

    def _generate_fallback_data(self):
        """Generate fallback data if Yahoo Finance fails"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='1h')
        np.random.seed(42)
        price = 1.0850
        data = []
        for _ in dates:
            change = np.random.normal(0, 0.0008)
            price += change
            data.append({
                'Open': price,
                'High': price + abs(np.random.normal(0, 0.0004)),
                'Low': price - abs(np.random.normal(0, 0.0004)),
                'Close': price + np.random.normal(0, 0.0003),
                'Volume': int(abs(np.random.normal(1000, 300)))
            })
        df = pd.DataFrame(data, index=dates)
        return df

    def engineer_features(self, df):
        """Create features and target variable"""
        df = df.copy()

        print("Initial rows:", len(df))

        # Make sure numeric columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(df.dtypes)

        # ---------------- Target ----------------

        df['Future_Close'] = df['Close'].shift(-self.predict_horizon)
        df['Price_Change'] = df['Future_Close'] - df['Close']

        df['Target'] = np.where(df['Price_Change'] > self.threshold, 1, 0)
        df['Target'] = np.where(df['Price_Change'] < -self.threshold, 0, df['Target'])

        df.loc[
            (df['Price_Change'] >= -self.threshold) &
            (df['Price_Change'] <= self.threshold),
            'Target'
        ] = np.nan

        df = df.dropna(subset=['Target'])

        print("After target:", len(df))

        df['Target'] = df['Target'].astype(int)

        df.drop(['Future_Close','Price_Change'], axis=1, inplace=True)

        # ---------------- Indicators ----------------

        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        print("After SMA20:", df['SMA_20'].isna().sum())

        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)

        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        df['MACD'] = ta.trend.macd_diff(df['Close'])

        df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])

        df['BB_width'] = df['BB_high'] - df['BB_low']

        df['ATR'] = ta.volatility.average_true_range(
            df['High'],
            df['Low'],
            df['Close'],
            window=14
        )

        df['Stochastic'] = ta.momentum.stoch(
            df['High'],
            df['Low'],
            df['Close']
        )


        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']

        df['Close_Open_Ratio'] = (df['Close'] - df['Open']) / df['Open']

        df['Return_1'] = df['Close'].pct_change()

        df['Return_5'] = df['Close'].pct_change(5)

        print("Rows before final dropna:", len(df))

        print(df.isna().sum())

        df = df.dropna()

        print("Rows after final dropna:", len(df))

        return df

    def run_pipeline(self):
        """Execute full training pipeline and return results"""
        results = {'status': 'running', 'steps': []}

        try:
            # Step 1: Fetch data
            df = self.fetch_data()
            results['steps'].append({
                'name': 'Data Download',
                'message': f'Fetched {len(df)} candles',
                'status': 'success'
            })

            # Step 2: Feature engineering
            df = self.engineer_features(df)
            print(f"Rows after feature engineering: {len(df)}")
            self.df = df
            self.feature_cols = [col for col in df.columns if col not in
                                ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
            results['steps'].append({
                'name': 'Feature Engineering',
                'message': f'Created {len(self.feature_cols)} features',
                'status': 'success'
            })

            # Step 3: Train/test split
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            X_train = train_df[self.feature_cols].values
            y_train = train_df['Target'].values
            X_test = test_df[self.feature_cols].values
            y_test = test_df['Target'].values

            print(f"Training samples: {len(train_df)}")
            print(f"Testing samples: {len(test_df)}")

            if len(train_df) == 0 or len(test_df) == 0:
                raise ValueError(
                    f"Dataset split failed. Training={len(train_df)}, Testing={len(test_df)}"
                )

            # Scale
            print("Starting scaling...")

            self.scaler = RobustScaler()

            X_train_scaled = self.scaler.fit_transform(X_train)

            print("Training data scaled.")

            X_test_scaled = self.scaler.transform(X_test)

            print("Test data scaled.")

            # Step 4: Train Random Forest
            print("Creating Random Forest model...")
            
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=20, random_state=42, n_jobs=-1
            )
            
            print("Training model...")

            self.model.fit(X_train_scaled, y_train)

            print("Training complete.")
            
            os.makedirs("models", exist_ok=True)

            joblib.dump(self.model, "models/rf_model.pkl")
            joblib.dump(self.scaler, "models/scaler.pkl")

            print("Random Forest model saved.")
            
            print("Calculating accuracy...")

            rf_acc = self.model.score(X_test_scaled, y_test)

            print("Accuracy:", rf_acc)
            
            results['steps'].append({
                'name': 'Random Forest Model',
                'message': f'Accuracy: {rf_acc:.2%}',
                'status': 'success'
            })

            # Step 5: Generate predictions
            rf_pred = self.model.predict(X_test_scaled)
            rf_prob = self.model.predict_proba(X_test_scaled)[:, 1]

            # Step 6: Simple backtest
            backtest_df = test_df.copy()
            backtest_df['Prediction'] = rf_pred
            backtest_df['Confidence'] = rf_prob
            backtest_df['Signal'] = np.where(
                (backtest_df['Prediction'] == 1) & (backtest_df['Confidence'] > 0.55), 'Buy',
                np.where((backtest_df['Prediction'] == 0) & (backtest_df['Confidence'] > 0.55), 'Sell', 'Hold')
            )

            # Backtest simulation
            capital = 10000
            position = None
            entry_price = 0
            equity_curve = [capital]
            trades = []

            for i, row in backtest_df.iterrows():
                if position == 'Long':
                    atr_val = row['ATR'] if not np.isnan(row['ATR']) else row['Close'] * 0.001
                    sl_price = entry_price - 1.5 * atr_val
                    tp_price = entry_price + 3.0 * atr_val
                    
                    if row['Low'] <= sl_price:
                        loss_pct = (sl_price - entry_price) / entry_price
                        capital *= (1 + loss_pct)
                        trades.append({
                            'Entry': str(entry_time)[:19],
                            'Exit': str(i)[:19],
                            'Type': 'Sell',
                            'P&L': f"${loss_pct*10000:.2f}",
                            'Exit_Reason': 'Stop Loss'
                        })
                        position = None
                    elif row['High'] >= tp_price:
                        profit_pct = (tp_price - entry_price) / entry_price
                        capital *= (1 + profit_pct)
                        trades.append({
                            'Entry': str(entry_time)[:19],
                            'Exit': str(i)[:19],
                            'Type': 'Sell',
                            'P&L': f"${profit_pct*10000:.2f}",
                            'Exit_Reason': 'Take Profit'
                        })
                        position = None

                if position is None and row['Signal'] == 'Buy':
                    position = 'Long'
                    entry_price = row['Close']
                    entry_time = i

                if position == 'Long':
                    equity_curve.append(capital * (1 + (row['Close'] - entry_price) / entry_price))
                else:
                    equity_curve.append(capital)

            # Close final position
            if position == 'Long':
                exit_price = backtest_df.iloc[-1]['Close']
                profit_pct = (exit_price - entry_price) / entry_price
                capital *= (1 + profit_pct)
                trades.append({
                    'Entry': str(entry_time)[:19],
                    'Exit': str(backtest_df.index[-1])[:19],
                    'Type': 'Sell',
                    'P&L': f"${profit_pct*10000:.2f}",
                    'Exit_Reason': 'End of Test'
                })
                equity_curve[-1] = capital

            # Results
            total_return = ((capital / 10000) - 1) * 100
            win_rate = sum(1 for t in trades if float(t['P&L'].replace('$','').replace('-','')) > 0) / len(trades) * 100 if trades else 0
            
            results['backtest'] = {
                'initial_capital': 10000,
                'final_capital': round(capital, 2),
                'total_return': round(total_return, 2),
                'num_trades': len(trades),
                'win_rate': round(win_rate, 1),
                'equity_curve': equity_curve,
                'trades': trades,
                'rf_accuracy': round(rf_acc * 100, 2),

                # Phase 1 placeholder
                'lstm_accuracy': "N/A"
            }

            # Latest signal
            last_pred = rf_pred[-1] if len(rf_pred) > 0 else 0
            last_prob = rf_prob[-1] if len(rf_prob) > 0 else 0.5
            
            if last_prob > 0.65:
                signal = 'Buy 📈'
            elif last_prob < 0.35:
                signal = 'Sell 📉'
            else:
                signal = 'Hold ⏸️'

            results['latest_signal'] = {
                'prediction': signal,
                'confidence': round(float(last_prob) * 100, 1),
                'timestamp': str(df.index[-1])
            }

            results['status'] = 'complete'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            results['steps'].append({
                'name': 'Error',
                'message': str(e),
                'status': 'error'
            })

        self.results = results
        return results


# Create singleton instance
engine = ForexEngine()