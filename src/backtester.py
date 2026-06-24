"""
回测框架模块 - 重构版
支持Walk-Forward验证、基准对比、风险指标

Author: jingke
Date: 2026-06-24
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger("quant_stock_picker")


@dataclass
class BacktestMetrics:
    """回测指标数据类"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    final_capital: float
    var_95: float  # 风险价值
    cvar_95: float  # 条件风险价值


class Backtester:
    """
    回测引擎
    
    功能:
    - 模拟交易执行
    - Walk-Forward交叉验证
    - 基准对比（买入持有/指数）
    - 风险指标（夏普、回撤、VaR、CVaR）
    """
    
    def __init__(self, initial_capital: float = 100000.0,
                 commission_rate: float = 0.0005,
                 slippage: float = 0.001) -> None:
        """
        初始化回测器
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（双边）
            slippage: 滑点
        """
        self.initial_capital: float = initial_capital
        self.commission_rate: float = commission_rate
        self.slippage: float = slippage
        
        # 状态变量
        self.capital: float = initial_capital
        self.position: float = 0.0
        self.trades: List[Dict] = []
        self.daily_values: List[float] = []
        
        logger.info(f"回测器初始化: 初始资金={initial_capital:,.0f}")
    
    def reset(self) -> None:
        """重置状态"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.trades = []
        self.daily_values = []
    
    def run(self, df: pd.DataFrame, signal_col: str = 'signal',
            price_col: str = 'close') -> pd.DataFrame:
        """
        运行回测
        
        Args:
            df: 包含价格和信号的DataFrame
            signal_col: 信号列名 (1=买入, 0=卖出/空仓)
            price_col: 价格列名
        
        Returns:
            包含回测结果的DataFrame
        """
        logger.info("开始回测...")
        self.reset()
        
        result_df = df.copy()
        portfolio_values = []
        
        for i in range(len(result_df)):
            current_price = result_df[price_col].iloc[i]
            signal = result_df[signal_col].iloc[i]
            
            # 执行交易
            if signal == 1 and self.position == 0:
                self._buy(current_price, result_df.index[i])
            elif signal == 0 and self.position > 0:
                self._sell(current_price, result_df.index[i])
            
            # 计算当前市值
            current_value = self._calculate_value(current_price)
            portfolio_values.append(current_value)
        
        result_df['portfolio_value'] = portfolio_values
        result_df['returns'] = result_df['portfolio_value'].pct_change()
        result_df['cumulative_return'] = (
            result_df['portfolio_value'] / self.initial_capital - 1
        )
        
        logger.info(f"回测完成: 最终市值={portfolio_values[-1]:,.2f}")
        logger.info(f"交易次数: {len(self.trades)}")
        
        return result_df
    
    def run_walk_forward(self, df: pd.DataFrame, model, 
                         feature_cols: List[str],
                         train_size: int = 252,
                         test_size: int = 63,
                         signal_threshold: float = 0.6) -> pd.DataFrame:
        """
        Walk-Forward交叉验证
        
        避免前视偏差，用历史数据训练，未来数据测试
        
        Args:
            df: 完整数据
            model: 模型实例
            feature_cols: 特征列
            train_size: 训练窗口大小（天数）
            test_size: 测试窗口大小（天数）
            signal_threshold: 信号阈值
        
        Returns:
            回测结果DataFrame
        """
        logger.info(f"开始Walk-Forward验证: train={train_size}, test={test_size}")
        
        self.reset()
        all_results = []
        
        n_samples = len(df)
        start_idx = 0
        window = 0
        
        while start_idx + train_size + test_size <= n_samples:
            window += 1
            train_start = start_idx
            train_end = start_idx + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            # 分割数据
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            
            # 训练
            X_train = train_df[feature_cols].values
            y_train = train_df['target'].values
            
            model.build_model()
            model.train(X_train, y_train)
            
            # 预测
            X_test = test_df[feature_cols].values
            _, y_prob = model.predict(X_test)
            
            # 生成信号
            test_df = test_df.copy()
            test_df['signal'] = (y_prob > signal_threshold).astype(int)
            
            # 回测这个窗口
            for i in range(len(test_df)):
                price = test_df['close'].iloc[i]
                signal = test_df['signal'].iloc[i]
                
                if signal == 1 and self.position == 0:
                    self._buy(price, test_df.index[i])
                elif signal == 0 and self.position > 0:
                    self._sell(price, test_df.index[i])
                
                value = self._calculate_value(price)
                self.daily_values.append(value)
            
            all_results.append(test_df)
            start_idx += test_size
            
            logger.info(f"  窗口 {window}: {train_start}-{test_end}")
            
            # 记录窗口结束时的市值
            final_price = test_df['close'].iloc[-1]
            final_value = self.capital + self.position * final_price
            logger.info(f"    窗口结束市值: {final_value:,.2f}")
        
        # 合并结果
        if all_results:
            result_df = pd.concat(all_results, ignore_index=True)
            result_df['portfolio_value'] = self.daily_values[:len(result_df)]
            result_df['returns'] = result_df['portfolio_value'].pct_change()
            result_df['cumulative_return'] = (
                result_df['portfolio_value'] / self.initial_capital - 1
            )
            return result_df
        else:
            return df.copy()
    
    def run_benchmark(self, df: pd.DataFrame, 
                     price_col: str = 'close') -> pd.DataFrame:
        """
        基准策略回测（买入持有）
        
        Args:
            df: 价格数据
            price_col: 价格列名
        
        Returns:
            基准回测结果
        """
        logger.info("计算基准策略（买入持有）...")
        
        result_df = df.copy()
        initial_price = result_df[price_col].iloc[0]
        
        # 买入持有：第一天全仓买入，持有到期
        result_df['benchmark_value'] = (
            result_df[price_col] / initial_price * self.initial_capital
        )
        result_df['benchmark_return'] = (
            result_df['benchmark_value'] / self.initial_capital - 1
        )
        
        return result_df
    
    def _buy(self, price: float, timestamp) -> None:
        """执行买入"""
        executed_price = price * (1 + self.slippage)
        max_amount = self.capital / (1 + self.commission_rate)
        shares = max_amount / executed_price
        
        cost = shares * executed_price
        commission = cost * self.commission_rate
        self.capital -= (cost + commission)
        self.position = shares
        
        self.trades.append({
            'type': 'buy',
            'timestamp': timestamp,
            'price': executed_price,
            'shares': shares,
            'cost': cost,
            'commission': commission,
            'capital_after': self.capital
        })
    
    def _sell(self, price: float, timestamp) -> None:
        """执行卖出"""
        executed_price = price * (1 - self.slippage)
        revenue = self.position * executed_price
        commission = revenue * self.commission_rate
        
        self.capital += (revenue - commission)
        self.position = 0.0
        
        self.trades.append({
            'type': 'sell',
            'timestamp': timestamp,
            'price': executed_price,
            'shares': self.position,
            'revenue': revenue,
            'commission': commission,
            'capital_after': self.capital
        })
    
    def _calculate_value(self, current_price: float) -> float:
        """计算当前总市值"""
        position_value = self.position * current_price
        return self.capital + position_value
    
    def calculate_metrics(self, result_df: pd.DataFrame) -> BacktestMetrics:
        """
        计算回测指标
        
        Args:
            result_df: 回测结果DataFrame
        
        Returns:
            BacktestMetrics指标对象
        """
        returns = result_df['returns'].dropna()
        portfolio_values = result_df['portfolio_value']
        
        # 基础收益指标
        total_return = (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        n_days = len(result_df)
        annualized_return = ((portfolio_values.iloc[-1] / self.initial_capital) ** 
                            (252 / n_days) - 1) * 100 if n_days > 0 else 0
        
        # 波动率
        volatility = returns.std() * np.sqrt(252) * 100
        
        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        if volatility > 0:
            sharpe_ratio = ((annualized_return / 100 - risk_free_rate) / 
                          (volatility / 100))
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        win_rate = (returns > 0).mean() * 100
        
        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        
        # VaR (95%)
        var_95 = float(np.percentile(returns.dropna(), 5) * 100)
        
        # CVaR (95%) - 条件风险价值
        cvar_95 = float(returns[returns <= np.percentile(returns.dropna(), 5)].mean() * 100)
        
        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_trades=len(self.trades),
            final_capital=portfolio_values.iloc[-1],
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        logger.info("回测指标:")
        logger.info(f"  总收益率: {metrics.total_return:.2f}%")
        logger.info(f"  年化收益率: {metrics.annualized_return:.2f}%")
        logger.info(f"  波动率: {metrics.volatility:.2f}%")
        logger.info(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  最大回撤: {metrics.max_drawdown:.2f}%")
        logger.info(f"  胜率: {metrics.win_rate:.2f}%")
        logger.info(f"  VaR(95%): {metrics.var_95:.2f}%")
        logger.info(f"  CVaR(95%): {metrics.cvar_95:.2f}%")
        
        return metrics
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        获取交易历史
        
        Returns:
            交易记录DataFrame
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
