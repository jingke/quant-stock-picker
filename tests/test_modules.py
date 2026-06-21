"""
测试模块
使用pytest进行单元测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src到路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model import StockPredictor
from backtester import Backtester


class TestDataFetcher:
    """测试数据获取模块"""
    
    def test_init(self):
        fetcher = DataFetcher(market='A_share', data_source='akshare')
        assert fetcher.market == 'A_share'
        assert fetcher.data_source == 'akshare'
    
    def test_mock_data(self):
        """测试模拟数据生成"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 101,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        assert len(df) == 100
        assert 'close' in df.columns


class TestFeatureEngineer:
    """测试特征工程模块"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        base = 100
        returns = np.random.randn(100) * 0.02
        close = base * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.randn(100) * 0.01),
            'high': close * (1 + abs(np.random.randn(100)) * 0.02),
            'low': close * (1 - abs(np.random.randn(100)) * 0.02),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        return df
    
    def test_add_technical_indicators(self, sample_data):
        engineer = FeatureEngineer(sample_data)
        df = engineer.add_technical_indicators()
        
        assert 'MA5' in df.columns
        assert 'MA20' in df.columns
        assert 'return_1d' in df.columns
        assert 'volume_ratio' in df.columns
    
    def test_add_target(self, sample_data):
        engineer = FeatureEngineer(sample_data)
        df = engineer.add_technical_indicators()
        df = engineer.add_target(horizon=5)
        
        assert 'future_return' in df.columns
        assert 'target' in df.columns
        assert df['target'].isin([0, 1]).all()


class TestStockPredictor:
    """测试模型模块"""
    
    @pytest.fixture
    def sample_xy(self):
        """生成样本数据"""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    def test_build_model(self):
        predictor = StockPredictor(model_type='random_forest')
        predictor.build_model()
        assert predictor.model is not None
    
    def test_train_and_predict(self, sample_xy):
        X, y = sample_xy
        predictor = StockPredictor(model_type='random_forest')
        predictor.build_model()
        predictor.train(X[:150], y[:150])
        
        y_pred, y_prob = predictor.predict(X[150:])
        assert len(y_pred) == 50
        assert len(y_prob) == 50
        assert all(p in [0, 1] for p in y_pred)
        assert all(0 <= p <= 1 for p in y_prob)
    
    def test_evaluate(self, sample_xy):
        X, y = sample_xy
        predictor = StockPredictor(model_type='random_forest')
        predictor.build_model()
        predictor.train(X[:150], y[:150])
        
        metrics = predictor.evaluate(X[150:], y[150:])
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestBacktester:
    """测试回测模块"""
    
    @pytest.fixture
    def backtest_data(self):
        """生成回测测试数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({
            'date': dates,
            'close': price,
            'signal': [1 if i % 10 < 5 else 0 for i in range(100)]
        })
        return df
    
    def test_init(self):
        backtester = Backtester(initial_capital=100000)
        assert backtester.initial_capital == 100000
        assert backtester.capital == 100000
    
    def test_run(self, backtest_data):
        backtester = Backtester(initial_capital=100000)
        result = backtester.run(backtest_data, signal_col='signal')
        
        assert 'portfolio_value' in result.columns
        assert len(result) == len(backtest_data)
        assert result['portfolio_value'].iloc[0] == 100000
    
    def test_calculate_metrics(self, backtest_data):
        backtester = Backtester(initial_capital=100000)
        result = backtester.run(backtest_data, signal_col='signal')
        metrics = backtester.calculate_metrics(result)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
