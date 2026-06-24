"""
数据获取模块 - 重构版
支持A股(AKShare)、美股(Yahoo Finance)等数据源

Author: jingke
Date: 2026-06-24
"""

import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class DataFetcher:
    """
    金融数据获取器
    
    支持多数据源:
    - A股: AKShare (免费，数据全)
    - 美股/港股: Yahoo Finance
    """
    
    def __init__(self, market: str = 'A_share', data_source: str = 'akshare') -> None:
        """
        初始化数据获取器
        
        Args:
            market: 市场类型 ('A_share', 'US', 'HK')
            data_source: 数据源 ('akshare', 'yfinance', 'tushare')
        """
        self.market: str = market
        self.data_source: str = data_source
        self._ak: Optional[Any] = None
        self._yf: Optional[Any] = None
        
        # 延迟导入，避免未安装报错
        if data_source == 'akshare':
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                logger.warning("akshare 未安装，尝试安装: pip install akshare")
                raise
        elif data_source == 'yfinance':
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.warning("yfinance 未安装，尝试安装: pip install yfinance")
                raise
        
        logger.info(f"数据获取器初始化: market={market}, source={data_source}")
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取个股历史K线数据
        
        Args:
            symbol: 股票代码
                - A股: '000001' (平安银行)
                - 美股: 'AAPL'
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
        
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume]
        
        Raises:
            ValueError: 不支持的数据源
            ConnectionError: 网络连接失败
        """
        logger.info(f"获取 {symbol} 数据: {start_date} - {end_date}")
        
        try:
            if self.data_source == 'akshare':
                return self._get_akshare_data(symbol, start_date, end_date)
            elif self.data_source == 'yfinance':
                return self._get_yfinance_data(symbol, start_date, end_date)
            else:
                raise ValueError(f"不支持的数据源: {self.data_source}")
        except Exception as e:
            logger.error(f"数据获取失败: {e}")
            raise ConnectionError(f"无法获取 {symbol} 数据: {e}")
    
    def _get_akshare_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """通过AKShare获取A股数据"""
        if self._ak is None:
            raise RuntimeError("AKShare未初始化")
        
        try:
            df = self._ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"AKShare获取成功: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"AKShare获取失败: {e}")
            raise
    
    def _get_yfinance_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """通过Yahoo Finance获取数据"""
        if self._yf is None:
            raise RuntimeError("Yahoo Finance未初始化")
        
        try:
            # 转换日期格式
            start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            ticker = self._yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            
            # 标准化列名
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Yahoo Finance获取成功: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"Yahoo Finance获取失败: {e}")
            raise
    
    def get_fundamental_data(self, symbol: str) -> pd.DataFrame:
        """
        获取财务指标数据
        
        Args:
            symbol: 股票代码
        
        Returns:
            DataFrame with fundamental indicators
        """
        logger.info(f"获取 {symbol} 财务数据")
        
        if self.data_source == 'akshare' and self._ak is not None:
            try:
                # 获取个股信息
                stock_info = self._ak.stock_individual_info_em(symbol=symbol)
                
                # 获取财务指标
                financial = self._ak.stock_financial_analysis_indicator(symbol=symbol)
                
                return financial
            except Exception as e:
                logger.warning(f"财务数据获取失败: {e}")
                return pd.DataFrame()
        else:
            logger.warning("当前数据源不支持财务数据获取")
            return pd.DataFrame()
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            DataFrame with stock list
        """
        if self.data_source == 'akshare' and self._ak is not None:
            return self._ak.stock_zh_a_spot_em()
        else:
            raise NotImplementedError("当前数据源不支持股票列表获取")
    
    def get_macro_data(self, indicator: str = 'cpi') -> pd.DataFrame:
        """
        获取宏观经济数据
        
        Args:
            indicator: 指标类型 ('cpi', 'ppi', 'pmi', 'interest_rate')
        
        Returns:
            DataFrame with macro data
        """
        logger.info(f"获取宏观经济数据: {indicator}")
        
        if self.data_source == 'akshare' and self._ak is not None:
            try:
                if indicator == 'cpi':
                    return self._ak.macro_china_cpi()
                elif indicator == 'ppi':
                    return self._ak.macro_china_ppi()
                elif indicator == 'pmi':
                    return self._ak.macro_china_pmi()
                else:
                    raise ValueError(f"不支持的宏观指标: {indicator}")
            except Exception as e:
                logger.warning(f"宏观数据获取失败: {e}")
                return pd.DataFrame()
        else:
            logger.warning("当前数据源不支持宏观数据获取")
            return pd.DataFrame()
