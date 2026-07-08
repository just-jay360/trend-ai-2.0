class TradingAssistant:

    def explain(self, signal, confidence, latest_row):

        reasons = []
        summary = []
        advice = []

        score = 0

        # EMA Analysis
        if latest_row["Close"] > latest_row["EMA_20"]:
            reasons.append("Price is above EMA20 (Bullish)")
            summary.append("Price is trading above the 20-period EMA, suggesting buyers currently have control.")
            score += 2
        else:
            reasons.append("Price is below EMA20 (Bearish)")
            summary.append("Price remains below the 20-period EMA, indicating the broader trend is bearish.")
            score -= 2

        # RSI Analysis
        if latest_row["RSI"] > 70:
            reasons.append("RSI is Overbought")
            summary.append("RSI is overbought, meaning bullish momentum may be weakening.")
            advice.append("Watch for a possible bearish reversal.")
            score -= 1

        elif latest_row["RSI"] < 30:
            reasons.append("RSI is Oversold")
            summary.append("RSI is oversold, suggesting selling pressure may be exhausted.")
            advice.append("Watch for a possible bullish reversal.")
            score += 1

        else:
            reasons.append("RSI is Neutral")
            summary.append("RSI is neutral, indicating balanced momentum.")

        # MACD Analysis
        if latest_row["MACD"] > 0:
            reasons.append("MACD Momentum is Bullish")
            summary.append("MACD remains positive, confirming bullish momentum.")
            score += 2
        else:
            reasons.append("MACD Momentum is Bearish")
            summary.append("MACD remains negative, confirming bearish momentum.")
            score -= 2

        # Market Bias
        if score >= 4:
            bias = "Strong Bullish"
        elif score >= 2:
            bias = "Bullish"
        elif score <= -4:
            bias = "Strong Bearish"
        elif score <= -2:
            bias = "Bearish"
        else:
            bias = "Neutral"

        # Risk
        if confidence >= 75:
            risk = "Low"
        elif confidence >= 55:
            risk = "Medium"
        else:
            risk = "High"

        # Trade Levels
        price = latest_row["Close"]

        if signal == "BUY":
            sl = round(price * 0.998, 5)
            tp = round(price * 1.004, 5)
            advice.append("Consider long entries while monitoring resistance.")

        elif signal == "SELL":
            sl = round(price * 1.002, 5)
            tp = round(price * 0.996, 5)
            advice.append("Consider short entries while monitoring support.")

        else:
            sl = "-"
            tp = "-"
            advice.append("Wait for stronger confirmation before opening a position.")

        return {
            "signal": signal,
            "confidence": confidence,
            "risk": risk,
            "market_bias": bias,
            "stop_loss": sl,
            "take_profit": tp,
            "reasons": reasons,
            "summary": " ".join(summary),
            "advice": advice
        }


assistant = TradingAssistant()