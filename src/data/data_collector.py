import os
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv

class CryptoDataCollector:
    """
    Data collector for cryptocurrency OHLCV data from various exchanges.
    """
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize the data collector.
        
        Args:
            exchange_name (str): Name of the exchange to collect data from
        """
        self.exchange = getattr(ccxt, exchange_name)()
        
        # Load API credentials if provided
        load_dotenv()
        api_key = os.getenv(f'{exchange_name.upper()}_API_KEY')
        api_secret = os.getenv(f'{exchange_name.upper()}_SECRET')
        
        if api_key and api_secret:
            self.exchange.apiKey = api_key
            self.exchange.secret = api_secret
    
    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a specific trading pair.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for the candles
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        # Convert dates to timestamps
        since = int(start_date.timestamp() * 1000)
        end = int(end_date.timestamp() * 1000)
        
        all_candles = []
        while since < end:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000  # Maximum number of candles per request
                )
                if not candles:
                    break
                    
                all_candles.extend(candles)
                since = candles[-1][0] + 1  # Next timestamp
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
                
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the OHLCV data for model training.
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            sequence_length (int): Number of time steps to use for each sample
            train_split (float): Proportion of data to use for training
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Create binary labels (1 for price up, 0 for price down)
        df['target'] = (df['returns'] > 0).astype(int)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:(i + sequence_length)][['open', 'high', 'low', 'close', 'volume']]
            target = df.iloc[i + sequence_length]['target']
            
            # Normalize sequence
            sequence = (sequence - sequence.mean()) / sequence.std()
            
            sequences.append(sequence.values)
            targets.append(target)
            
        X = np.array(sequences)
        y = np.array(targets)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test 