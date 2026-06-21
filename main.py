"""
Quant Stock Picker - Main Entry Point
基于机器学习的A股智能选股系统主入口

Usage:
    python main.py --config config/config.yaml
    python main.py --mode train --config config/config.yaml
    python main.py --mode backtest --config config/config.yaml
    python main.py --mode predict --symbol 000001 --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import pandas as pd
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from model import StockPredictor
from backtester import Backtester
from evaluator import Evaluator


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger("quant_stock_picker")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_data_pipeline(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    数据获取和特征工程流水线
    
    Args:
        config: 配置字典
        logger: 日志对象
    
    Returns:
        处理后的DataFrame
    """
    logger.info("=" * 50)
    logger.info("开始数据获取和特征工程")
    logger.info("=" * 50)
    
    # 初始化数据获取器
    data_config = config.get('data', {})
    fetcher = DataFetcher(
        market=data_config.get('market', 'A_share'),
        data_source=data_config.get('source', 'akshare')
    )
    
    # 获取股票列表
    symbols = data_config.get('symbols', ['000001'])  # 默认平安银行
    logger.info(f"目标股票: {symbols}")
    
    # 计算时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=data_config.get('lookback_days', 730))
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    logger.info(f"数据时间范围: {start_str} - {end_str}")
    
    # 获取数据
    all_data = []
    for symbol in symbols:
        logger.info(f"正在获取 {symbol} 数据...")
        try:
            df = fetcher.get_stock_data(symbol, start_str, end_str)
            df['symbol'] = symbol
            all_data.append(df)
            logger.info(f"  ✓ 获取成功: {len(df)} 条记录")
        except Exception as e:
            logger.error(f"  ✗ 获取失败: {e}")
    
    if not all_data:
        raise ValueError("没有成功获取任何股票数据")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"合并数据: {len(combined_df)} 条记录")
    
    # 特征工程
    logger.info("开始特征工程...")
    engineer = FeatureEngineer(combined_df)
    
    # 添加技术指标
    df_with_tech = engineer.add_technical_indicators()
    logger.info("  ✓ 技术指标添加完成")
    
    # 添加财务因子（如果有）
    try:
        df_with_fund = engineer.add_fundamental_features()
        logger.info("  ✓ 财务因子添加完成")
    except Exception as e:
        logger.warning(f"  ! 财务因子添加失败（可能无数据）: {e}")
        df_with_fund = df_with_tech
    
    # 添加目标变量
    horizon = config.get('model', {}).get('prediction_horizon', 5)
    df_final = engineer.add_target(horizon=horizon)
    logger.info(f"  ✓ 目标变量添加完成（预测周期: {horizon}天）")
    
    # 删除NaN值
    df_clean = df_final.dropna()
    logger.info(f"清洗后数据: {len(df_clean)} 条记录")
    
    # 保存处理后的数据
    output_dir = Path(config.get('output', {}).get('data_dir', 'data/processed'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"数据已保存: {output_path}")
    
    return df_clean


def run_training(df: pd.DataFrame, config: dict, logger: logging.Logger) -> StockPredictor:
    """
    模型训练
    
    Args:
        df: 特征工程后的数据
        config: 配置字典
        logger: 日志对象
    
    Returns:
        训练好的模型
    """
    logger.info("=" * 50)
    logger.info("开始模型训练")
    logger.info("=" * 50)
    
    model_config = config.get('model', {})
    
    # 定义特征列（排除非特征列）
    exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 
                    'future_return', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"特征列表: {feature_cols}")
    
    # 准备数据
    X = df[feature_cols].values
    y = df['target'].values
    
    # 时间序列分割（避免数据泄露）
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"训练集: {len(X_train)} 条")
    logger.info(f"测试集: {len(X_test)} 条")
    logger.info(f"训练集正样本比例: {y_train.mean():.2%}")
    logger.info(f"测试集正样本比例: {y_test.mean():.2%}")
    
    # 初始化模型
    model_type = model_config.get('type', 'xgboost')
    logger.info(f"模型类型: {model_type}")
    
    predictor = StockPredictor(model_type=model_type)
    predictor.feature_names = feature_cols
    predictor.build_model()
    
    # 训练
    logger.info("开始训练...")
    predictor.train(X_train, y_train)
    logger.info("训练完成")
    
    # 评估
    logger.info("模型评估:")
    metrics = predictor.evaluate(X_test, y_test)
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # 特征重要性
    importance = predictor.feature_importance()
    logger.info("特征重要性 Top 10:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_importance[:10]:
        logger.info(f"  {feature}: {imp:.4f}")
    
    # 保存模型
    model_dir = Path(config.get('output', {}).get('model_dir', 'models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{model_type}_{datetime.now().strftime('%Y%m%d')}.json"
    predictor.save(str(model_path))
    logger.info(f"模型已保存: {model_path}")
    
    # 保存特征重要性图
    fig_dir = Path(config.get('output', {}).get('figure_dir', 'results/figures'))
    fig_dir.mkdir(parents=True, exist_ok=True)
    Evaluator.plot_feature_importance(
        importance, 
        save_path=str(fig_dir / "feature_importance.png")
    )
    logger.info("特征重要性图已保存")
    
    return predictor


def run_backtest(df: pd.DataFrame, predictor: StockPredictor, 
                 config: dict, logger: logging.Logger) -> dict:
    """
    策略回测
    
    Args:
        df: 完整数据
        predictor: 训练好的模型
        config: 配置字典
        logger: 日志对象
    
    Returns:
        回测指标字典
    """
    logger.info("=" * 50)
    logger.info("开始策略回测")
    logger.info("=" * 50)
    
    # 准备特征
    exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 
                    'future_return', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    # 生成预测信号
    logger.info("生成交易信号...")
    _, y_prob = predictor.predict(X)
    
    # 构建信号：概率 > 阈值则买入
    threshold = config.get('backtest', {}).get('signal_threshold', 0.6)
    df['signal'] = 0
    df.loc[y_prob > threshold, 'signal'] = 1
    
    logger.info(f"信号阈值: {threshold}")
    logger.info(f"买入信号次数: {(df['signal'] == 1).sum()}")
    logger.info(f"卖出/空仓次数: {(df['signal'] == 0).sum()}")
    
    # 运行回测
    initial_capital = config.get('backtest', {}).get('initial_capital', 100000)
    backtester = Backtester(initial_capital=initial_capital)
    
    logger.info(f"初始资金: {initial_capital:,.0f}")
    result_df = backtester.run(df)
    
    # 计算指标
    metrics = backtester.calculate_metrics(result_df)
    
    logger.info("回测结果:")
    logger.info(f"  总收益率: {metrics['total_return']:.2f}%")
    logger.info(f"  年化收益率: {metrics['annualized_return']:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {metrics['win_rate']:.2f}%")
    
    # 保存回测结果
    fig_dir = Path(config.get('output', {}).get('figure_dir', 'results/figures'))
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成回测图表
    benchmark_df = df[['close']].copy()
    Evaluator.plot_backtest(
        result_df, 
        benchmark_df=benchmark_df,
        save_path=str(fig_dir / "backtest_result.png")
    )
    logger.info("回测图表已保存")
    
    # 保存回测报告
    report_dir = Path(config.get('output', {}).get('report_dir', 'results'))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d')}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Quant Stock Picker - Backtest Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("回测参数:\n")
        f.write(f"  初始资金: {initial_capital:,.0f}\n")
        f.write(f"  信号阈值: {threshold}\n\n")
        f.write("回测结果:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    logger.info(f"回测报告已保存: {report_path}")
    
    return metrics


def run_prediction(symbol: str, config: dict, logger: logging.Logger) -> dict:
    """
    对单只股票进行预测
    
    Args:
        symbol: 股票代码
        config: 配置字典
        logger: 日志对象
    
    Returns:
        预测结果字典
    """
    logger.info("=" * 50)
    logger.info(f"开始预测: {symbol}")
    logger.info("=" * 50)
    
    # 加载最新模型
    model_dir = Path(config.get('output', {}).get('model_dir', 'models'))
    model_files = list(model_dir.glob("*.json"))
    if not model_files:
        raise FileNotFoundError("没有找到训练好的模型，请先运行训练模式")
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"加载模型: {latest_model}")
    
    predictor = StockPredictor()
    predictor.load(str(latest_model))
    
    # 获取最新数据
    fetcher = DataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 取最近60天用于计算指标
    
    df = fetcher.get_stock_data(symbol, start_date.strftime('%Y%m%d'), 
                                end_date.strftime('%Y%m%d'))
    
    # 特征工程
    engineer = FeatureEngineer(df)
    df = engineer.add_technical_indicators()
    df = df.dropna()
    
    if len(df) == 0:
        raise ValueError("特征工程后数据为空")
    
    # 取最新一条进行预测
    latest_data = df.iloc[-1:]
    exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = latest_data[feature_cols].values
    
    prediction, probability = predictor.predict(X)
    
    result = {
        'symbol': symbol,
        'date': latest_data['date'].iloc[0],
        'current_price': latest_data['close'].iloc[0],
        'prediction': int(prediction[0]),
        'probability': float(probability[0]),
        'signal': 'BUY' if prediction[0] == 1 else 'HOLD/SELL'
    }
    
    logger.info(f"预测结果:")
    logger.info(f"  当前价格: {result['current_price']:.2f}")
    logger.info(f"  预测信号: {result['signal']}")
    logger.info(f"  上涨概率: {result['probability']:.2%}")
    
    return result


def main():
    """主入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Quant Stock Picker - 智能量化选股系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --config config/config.yaml
    python main.py --mode train --config config/config.yaml
    python main.py --mode backtest --config config/config.yaml
    python main.py --mode predict --symbol 000001 --config config/config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='配置文件路径 (默认: config/config.yaml)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'data', 'train', 'backtest', 'predict'],
        default='full',
        help='运行模式: full=完整流程, data=仅数据, train=仅训练, backtest=仅回测, predict=预测 (默认: full)'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='000001',
        help='预测目标股票代码 (默认: 000001 平安银行)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 50)
    logger.info("Quant Stock Picker 启动")
    logger.info("=" * 50)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"配置文件: {args.config}")
    
    try:
        # 加载配置
        config = load_config(args.config)
        logger.info("配置加载成功")
        
        # 根据模式执行
        if args.mode in ['full', 'data']:
            df = run_data_pipeline(config, logger)
        
        if args.mode in ['full', 'train']:
            if 'df' not in locals():
                # 尝试加载已有数据
                data_dir = Path(config.get('output', {}).get('data_dir', 'data/processed'))
                data_files = list(data_dir.glob("processed_data_*.csv"))
                if not data_files:
                    raise FileNotFoundError("没有找到数据，请先运行 data 模式")
                latest_data = max(data_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"加载已有数据: {latest_data}")
                df = pd.read_csv(latest_data)
            
            predictor = run_training(df, config, logger)
        
        if args.mode in ['full', 'backtest']:
            if 'df' not in locals():
                raise ValueError("回测需要数据，请先运行 data 或 train 模式")
            if 'predictor' not in locals():
                # 尝试加载已有模型
                model_dir = Path(config.get('output', {}).get('model_dir', 'models'))
                model_files = list(model_dir.glob("*.json"))
                if not model_files:
                    raise FileNotFoundError("没有找到模型，请先运行 train 模式")
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                predictor = StockPredictor()
                predictor.load(str(latest_model))
                logger.info(f"加载已有模型: {latest_model}")
            
            run_backtest(df, predictor, config, logger)
        
        if args.mode == 'predict':
            result = run_prediction(args.symbol, config, logger)
            logger.info(f"预测完成: {result}")
        
        logger.info("=" * 50)
        logger.info("运行完成")
        logger.info("=" * 50)
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        logger.error("请检查配置文件路径或先运行前置步骤")
        sys.exit(1)
    except Exception as e:
        logger.error(f"运行错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
