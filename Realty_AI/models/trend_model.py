import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TrendModel:
    def __init__(self):
        pass
        
    def predict(self, start_date=datetime.now(), periods=365):
        """
        Generates synthetic trend data.
        """
        dates = [start_date + timedelta(days=x) for x in range(periods)]
        
        # Trend + Seasonality + Noise
        t = np.linspace(0, 4 * np.pi, periods)
        trend = np.linspace(6000000, 8000000, periods) # Upward trend (60L to 80L)
        seasonality = 200000 * np.sin(t)
        
        # Generate OHLC
        closes = trend + seasonality + np.random.normal(0, 50000, periods)
        opens = np.roll(closes, 1)
        opens[0] = closes[0] - np.random.normal(0, 50000)
        
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 30000, periods))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 30000, periods))
        
        df = pd.DataFrame({
            'Date': dates, 
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes
        })
        return df
