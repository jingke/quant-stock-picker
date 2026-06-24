"""
仓位管理模块
支持Kelly公式、等权重、固定比例等策略

Author: jingke
Date: 2026-06-24
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class PositionSizingMethod(Enum):
    """仓位管理方法枚举"""
    EQUAL_WEIGHT = "equal_weight"          # 等权重
    KELLY_CRITERION = "kelly"              # Kelly公式
    FIXED_RATIO = "fixed_ratio"            # 固定比例
    VOLATILITY_TARGET = "vol_target"       # 波动率目标


class PositionSizer:
    """
    仓位管理器
    
    根据风险模型决定每次交易的仓位大小
    """
    
    def __init__(self, method: PositionSizingMethod = PositionSizingMethod.EQUAL_WEIGHT,
                 max_position: float = 1.0) -> None:
        """
        初始化仓位管理器
        
        Args:
            method: 仓位管理方法
            max_position: 最大仓位比例（0-1）
        """
        self.method: PositionSizingMethod = method
        self.max_position: float = max_position
        
        logger.info(f"仓位管理器初始化: method={method.value}, max_position={max_position}")
    
    def calculate_position(self, capital: float, price: float,
                          win_rate: Optional[float] = None,
                          avg_win: Optional[float] = None,
                          avg_loss: Optional[float] = None,
                          volatility: Optional[float] = None) -> float:
        """
        计算仓位大小
        
        Args:
            capital: 当前资金
            price: 当前价格
            win_rate: 胜率（Kelly需要）
            avg_win: 平均盈利（Kelly需要）
            avg_loss: 平均亏损（Kelly需要）
            volatility: 波动率（波动率目标需要）
        
        Returns:
            应买入的股数
        """
        if self.method == PositionSizingMethod.EQUAL_WEIGHT:
            return self._equal_weight(capital, price)
        
        elif self.method == PositionSizingMethod.KELLY_CRITERION:
            if win_rate is None or avg_win is None or avg_loss is None:
                logger.warning("Kelly公式需要win_rate, avg_win, avg_loss，回退到等权重")
                return self._equal_weight(capital, price)
            return self._kelly_criterion(capital, price, win_rate, avg_win, avg_loss)
        
        elif self.method == PositionSizingMethod.FIXED_RATIO:
            return self._fixed_ratio(capital, price)
        
        elif self.method == PositionSizingMethod.VOLATILITY_TARGET:
            if volatility is None:
                logger.warning("波动率目标需要volatility，回退到等权重")
                return self._equal_weight(capital, price)
            return self._volatility_target(capital, price, volatility)
        
        else:
            raise ValueError(f"不支持的仓位管理方法: {self.method}")
    
    def _equal_weight(self, capital: float, price: float) -> float:
        """
        等权重策略：全仓买入
        
        Args:
            capital: 当前资金
            price: 当前价格
        
        Returns:
            股数
        """
        # 使用90%资金，保留10%现金
        position_size = capital * 0.9 * self.max_position
        shares = position_size / price
        
        logger.debug(f"等权重: 资金={capital:.2f}, 价格={price:.2f}, 股数={shares:.2f}")
        return shares
    
    def _kelly_criterion(self, capital: float, price: float,
                         win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly公式：f = (bp - q) / b
        
        其中:
        - b = 平均盈利/平均亏损（盈亏比）
        - p = 胜率
        - q = 败率 = 1 - p
        
        Args:
            capital: 当前资金
            price: 当前价格
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
        
        Returns:
            股数
        """
        if avg_loss <= 0:
            logger.warning("平均亏损必须>0，回退到等权重")
            return self._equal_weight(capital, price)
        
        b = avg_win / avg_loss  # 盈亏比
        p = win_rate
        q = 1 - p
        
        # Kelly比例
        kelly_f = (b * p - q) / b
        
        # 限制在合理范围（0.1 - 0.5）
        kelly_f = max(0.1, min(0.5, kelly_f))
        
        position_size = capital * kelly_f * self.max_position
        shares = position_size / price
        
        logger.info(f"Kelly: f={kelly_f:.2%}, 资金={capital:.2f}, 股数={shares:.2f}")
        return shares
    
    def _fixed_ratio(self, capital: float, price: float) -> float:
        """
        固定比例策略：每次使用固定百分比资金
        
        Args:
            capital: 当前资金
            price: 当前价格
        
        Returns:
            股数
        """
        fixed_ratio = 0.2  # 每次使用20%资金
        
        position_size = capital * fixed_ratio * self.max_position
        shares = position_size / price
        
        logger.debug(f"固定比例: 资金={capital:.2f}, 比例={fixed_ratio:.0%}, 股数={shares:.2f}")
        return shares
    
    def _volatility_target(self, capital: float, price: float,
                          volatility: float) -> float:
        """
        波动率目标策略：根据波动率调整仓位
        
        波动率越高，仓位越小
        
        Args:
            capital: 当前资金
            price: 当前价格
            volatility: 年化波动率
        
        Returns:
            股数
        """
        target_vol = 0.15  # 目标年化波动率15%
        
        if volatility <= 0:
            logger.warning("波动率必须>0，回退到等权重")
            return self._equal_weight(capital, price)
        
        # 根据波动率调整仓位
        vol_ratio = target_vol / volatility
        vol_ratio = min(1.0, vol_ratio)  # 最大100%
        
        position_size = capital * vol_ratio * self.max_position
        shares = position_size / price
        
        logger.info(f"波动率目标: 波动率={volatility:.2%}, 目标={target_vol:.2%}, "
                   f"比例={vol_ratio:.2%}, 股数={shares:.2f}")
        return shares
    
    def get_method_name(self) -> str:
        """
        获取方法名称
        
        Returns:
            方法名称
        """
        return self.method.value
