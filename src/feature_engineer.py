"""
特征工程模块
构建技术指标、财务因子和宏观因子
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class FeatureEngineer:
    """
    特征工程器
    
    构建多维度特征:
    - 技术指标: MA, RSI, MACD, 布林带
    - 财务因子: PE, PB, ROE
    - 目标变量: 未来N日收益率
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化特征工程器
        
        Args:
            df: 原始数据DataFrame
        """
        self.df = df.copy()
        logger.info(f"特征工程器初始化: {len(df)} 条记录")
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        添加技术指标
        
        Returns:
            添加技术指标后的DataFrame
        """
        logger.info("添加技术指标...")
        
        # 确保数据按日期排序
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 移动平均线
        self.df['MA5'] = self.df['close'].rolling(window=5).mean()
        self.df['MA10'] = self.df['close'].rolling(window=10).mean()
        self.df['MA20'] = self.df['close'].rolling(window=20).mean()
        self.df['MA60'] = self.df['close'].rolling(window=60).mean()
        
        # 价格与均线关系
        self.df['close_to_MA5'] = self.df['close'] / self.df['MA5'] - 1
        self.df['close_to_MA20'] = self.df['close'] / self.df['MA20'] - 1
        self.df['MA5_to_MA20'] = self.df['MA5'] / self.df['MA20'] - 1
        
        # 波动率指标
        self.df['volatility_5'] = self.df['close'].rolling(window=5).std()
        self.df['volatility_20'] = self.df['close'].rolling(window=20).std()
        
        # 涨跌幅统计
        self.df['return_1d'] = self.df['close'].pct_change(1)
        self.df['return_5d'] = self.df['close'].pct_change(5)
        self.df['return_20d'] = self.df['close'].pct_change(20)
        
        # 价格位置
        self.df['high_20d'] = self.df['high'].rolling(window=20).max()
        self.df['low_20d'] = self.df['low'].rolling(window=20).min()
        self.df['price_position'] = (self.df['close'] - self.df['low_20d']) / \
                                     (self.df['high_20d'] - self.df['low_20d'] + 1e-10)
        
        # 成交量指标
        self.df['volume_MA5'] = self.df['volume'].rolling(window=5).mean()
        self.df['volume_MA20'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_MA20']
        
        # 尝试使用TA-Lib（如果安装了）
        try:
            import talib
            
            # RSI
            self.df['RSI'] = talib.RSI(self.df['close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                self.df['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            self.df['MACD'] = macd
            self.df['MACD_signal'] = macd_signal
            self.df['MACD_hist'] = macd_hist
            
            # 布林带
            upper, middle, lower = talib.BBANDS(
                self.df['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            self.df['BB_upper'] = upper
            self.df['BB_middle'] = middle
            self.df['BB_lower'] = lower
            self.df['BB_position'] = (self.df['close'] - lower) / (upper - lower + 1e-10)
            
            # KDJ
            k, d = talib.STOCH(
                self.df['high'], 
                self.df['low'], 
                self.df['close'],
                fastk_period=9, 
                slowk_period=3, 
                slowd_period=3
            )
            self.df['KDJ_K'] = k
            self.df['KDJ_D'] = d
            
            logger.info("  ✓ TA-Lib技术指标添加完成")
            
        except ImportError:
            logger.warning("  ! TA-Lib未安装，跳过高级技术指标")
            logger.warning("  安装命令: pip install TA-Lib")
        
        logger.info(f"技术指标添加完成，当前列数: {len(self.df.columns)}")
        return self.df
    
    def add_fundamental_features(self, fundamental_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        添加财务因子
        
        Args:
            fundamental_df: 财务数据DataFrame（可选）
        
        Returns:
            添加财务因子后的DataFrame
        """
        logger.info("添加财务因子...")
        
        # 基于价格数据计算的基础财务指标
        # 市净率近似（假设每股净资产为过去250日均价的0.5倍）
        # 实际应用中应从财务数据获取
        self.df['book_value_approx'] = self.df['close'].rolling(window=250).mean() * 0.5
        self.df['PB_approx'] = self.df['close'] / (self.df['book_value_approx'] + 1e-10)
        
        # 市盈率近似（假设每股收益为过去250日均价*0.05）
        self.df['eps_approx'] = self.df['close'].rolling(window=250).mean() * 0.05
        self.df['PE_approx'] = self.df['close'] / (self.df['eps_approx'] + 1e-10)
        
        # 营收增长率近似（基于价格动量）
        self.df['revenue_growth_approx'] = self.df['close'].pct_change(60)
        
        # ROE近似（基于价格动量和波动率）
        self.df['roe_approx'] = self.df['return_20d'] / (self.df['volatility_20'] + 1e-10)
        
        # 如果提供了外部财务数据，合并
        if fundamental_df is not None and not fundamental_df.empty:
            logger.info("  合并外部财务数据")
            # 这里可以添加合并逻辑
            # 例如：self.df = self.df.merge(fundamental_df, on='date', how='left')
        
        logger.info(f"财务因子添加完成，当前列数: {len(self.df.columns)}")
        return self.df
    
    def add_macro_features(self, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        添加宏观因子
        
        Args:
            macro_df: 宏观数据DataFrame（可选）
        
        Returns:
            添加宏观因子后的DataFrame
        """
        logger.info("添加宏观因子...")
        
        if macro_df is not None and not macro_df.empty:
            # 合并宏观数据
            logger.info("  使用外部宏观数据")
            # 按日期合并的逻辑
        
        logger.info(f"宏观因子添加完成，当前列数: {len(self.df.columns)}")
        return self.df
    
    def add_target(self, horizon: int = 5) -> pd.DataFrame:
        """
        添加目标变量
        
        Args:
            horizon: 预测周期（天数）
        
        Returns:
            添加目标变量后的DataFrame
        """
        logger.info(f"添加目标变量（预测周期: {horizon}天）...")
        
        # 未来N日收益率
        future_close = self.df['close'].shift(-horizon)
        self.df['future_return'] = (future_close - self.df['close']) / self.df['close']
        
        # 分类目标：未来N日是否上涨
        self.df['target'] = (self.df['future_return'] > 0).astype(int)
        
        # 回归目标（可选）
        self.df['target_return'] = self.df['future_return']
        
        logger.info(f"目标变量添加完成")
        logger.info(f"  正样本比例: {self.df['target'].mean():.2%}")
        
        return self.df
    
    def get_feature_columns(self) -> list:
        """
        获取特征列名列表（排除非特征列）
        
        Returns:
            特征列名列表
        """
        exclude_cols = [
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'amount', 'amplitude', 'pct_change', 'change', 'turnover',
            'future_return', 'target', 'target_return'
        ]
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_train_data(self, drop_na: bool = True) -> pd.DataFrame:
        """
        准备训练数据
        
        Args:
            drop_na: 是否删除NaN值
        
        Returns:
            处理后的DataFrame
        """
        if drop_na:
            original_len = len(self.df)
            self.df = self.df.dropna()
            dropped = original_len - len(self.df)
            logger.info(f"删除NaN值: {dropped} 条记录被删除，剩余 {len(self.df)} 条")
        
        return self.df
