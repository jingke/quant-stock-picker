"""
Demo脚本：生成示例回测结果和图表
用于README展示和项目验证
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from feature_engineer import FeatureEngineer
from model import StockPredictor
from backtester import Backtester
from evaluator import Evaluator


def generate_mock_data(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成模拟股票数据"""
    np.random.seed(seed)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # 生成随机游走价格
    returns = np.random.randn(n_days) * 0.02
    close = 100 * (1 + returns).cumprod()
    
    # 生成OHLCV
    df = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(n_days) * 0.005),
        'high': close * (1 + abs(np.random.randn(n_days)) * 0.015),
        'low': close * (1 - abs(np.random.randn(n_days)) * 0.015),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    return df


def run_demo():
    """运行完整演示"""
    print("=" * 60)
    print("Quant Stock Picker - Demo")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1/5] 生成模拟数据...")
    df = generate_mock_data(n_days=500)
    print(f"  生成 {len(df)} 天数据")
    
    # 2. 特征工程
    print("\n[2/5] 特征工程...")
    engineer = FeatureEngineer(df)
    df = engineer.add_technical_indicators()
    df = engineer.add_target(horizon=5)
    df = df.dropna()
    print(f"  处理后数据: {len(df)} 条")
    print(f"  特征数量: {len(engineer.get_feature_columns())}")
    
    # 3. 训练模型
    print("\n[3/5] 训练模型...")
    feature_cols = engineer.get_feature_columns()
    X = df[feature_cols].values
    y = df['target'].values
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    predictor = StockPredictor(model_type='xgboost')
    predictor.feature_names = feature_cols
    predictor.build_model()
    predictor.train(X_train, y_train)
    
    metrics = predictor.evaluate(X_test, y_test)
    print(f"  测试集准确率: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    # 4. 回测
    print("\n[4/5] 策略回测...")
    _, y_prob = predictor.predict(X)
    df['signal'] = (y_prob > 0.6).astype(int)
    
    backtester = Backtester(initial_capital=100000)
    result_df = backtester.run(df, signal_col='signal')
    backtest_metrics = backtester.calculate_metrics(result_df)
    
    print(f"  总收益率: {backtest_metrics['total_return']:.2f}%")
    print(f"  年化收益: {backtest_metrics['annualized_return']:.2f}%")
    print(f"  夏普比率: {backtest_metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {backtest_metrics['max_drawdown']:.2f}%")
    
    # 5. 生成图表
    print("\n[5/5] 生成图表...")
    
    # 创建输出目录
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 回测图表
    Evaluator.plot_backtest(
        result_df,
        save_path=str(output_dir / 'backtest_demo.png')
    )
    print(f"  回测图表: {output_dir / 'backtest_demo.png'}")
    
    # 特征重要性
    importance = predictor.feature_importance()
    Evaluator.plot_feature_importance(
        importance,
        save_path=str(output_dir / 'feature_importance_demo.png'),
        top_n=10
    )
    print(f"  特征重要性: {output_dir / 'feature_importance_demo.png'}")
    
    # 生成报告
    report = Evaluator.generate_report(
        result_df,
        backtest_metrics,
        save_path='results/demo_report.txt'
    )
    
    print("\n" + "=" * 60)
    print("Demo完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - results/figures/backtest_demo.png")
    print("  - results/figures/feature_importance_demo.png")
    print("  - results/demo_report.txt")
    
    return result_df, backtest_metrics


if __name__ == '__main__':
    run_demo()
