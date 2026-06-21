"""
回测框架模块
模拟真实交易，计算策略收益和风险指标
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class Backtester:
    """
    回测引擎
    
    功能:
    - 模拟交易执行
    - 计算收益曲线
    - 评估风险指标（夏普比率、最大回撤等）
    """
    
    def __init__(self, initial_capital: float = 100000.0,
                 commission_rate: float = 0.0005,
                 slippage: float = 0.001):
        """
        初始化回测器
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（双边）
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # 状态变量
        self.capital = initial_capital
        self.position = 0.0  # 持仓数量
        self.trades = []  # 交易记录
        self.daily_values = []  # 每日市值
        
        logger.info(f"回测器初始化: 初始资金={initial_capital:,.0f}")
    
    def reset(self):
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
                # 买入信号且空仓 -> 全仓买入
                self._buy(current_price, result_df.index[i])
                
            elif signal == 0 and self.position > 0:
                # 卖出信号且持仓 -> 全部卖出
                self._sell(current_price, result_df.index[i])
            
            # 计算当前市值
            current_value = self._calculate_value(current_price)
            portfolio_values.append(current_value)
        
        result_df['portfolio_value'] = portfolio_values
        result_df['returns'] = result_df['portfolio_value'].pct_change()
        
        # 计算累计收益
        result_df['cumulative_return'] = (
            result_df['portfolio_value'] / self.initial_capital - 1
        )
        
        logger.info(f"回测完成: 最终市值={portfolio_values[-1]:,.2f}")
        logger.info(f"交易次数: {len(self.trades)}")
        
        return result_df
    
    def _buy(self, price: float, timestamp):
        """执行买入"""
        # 考虑滑点
        executed_price = price * (1 + self.slippage)
        
        # 计算可买入数量（扣除手续费）
        max_amount = self.capital / (1 + self.commission_rate)
        shares = max_amount / executed_price
        
        # 更新状态
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
        
        logger.debug(f"买入: 价格={executed_price:.2f}, 数量={shares:.2f}")
    
    def _sell(self, price: float, timestamp):
        """执行卖出"""
        # 考虑滑点
        executed_price = price * (1 - self.slippage)
        
        # 计算收入
        revenue = self.position * executed_price
        commission = revenue * self.commission_rate
        
        # 更新状态
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
        
        logger.debug(f"卖出: 价格={executed_price:.2f}, 收入={revenue:.2f}")
    
    def _calculate_value(self, current_price: float) -> float:
        """计算当前总市值"""
        position_value = self.position * current_price
        return self.capital + position_value
    
    def calculate_metrics(self, result_df: pd.DataFrame) -> dict:
        """
        计算回测指标
        
        Args:
            result_df: 回测结果DataFrame
        
        Returns:
            指标字典
        """
        returns = result_df['returns'].dropna()
        portfolio_values = result_df['portfolio_value']
        
        # 基础收益指标
        total_return = (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益（假设252个交易日）
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
        
        # 交易统计
        n_trades = len(self.trades)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': n_trades,
            'final_capital': portfolio_values.iloc[-1]
        }
        
        logger.info("回测指标:")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {name}: {value:.2f}")
            else:
                logger.info(f"  {name}: {value}")
        
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
