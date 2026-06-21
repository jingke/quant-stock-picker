"""
可视化评估模块
生成回测图表和特征重要性图
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class Evaluator:
    """
    评估可视化器
    
    功能:
    - 回测结果可视化
    - 特征重要性图
    - 收益曲线、回撤图
    """
    
    @staticmethod
    def plot_backtest(result_df: pd.DataFrame, 
                     benchmark_df: Optional[pd.DataFrame] = None,
                     save_path: Optional[str] = None) -> str:
        """
        绘制回测结果图
        
        Args:
            result_df: 回测结果DataFrame
            benchmark_df: 基准数据（可选）
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.error("matplotlib未安装，请运行: pip install matplotlib")
            raise
        
        logger.info("生成回测图表...")
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. 累计收益曲线
        ax1 = axes[0]
        ax1.plot(result_df.index, result_df['portfolio_value'] / result_df['portfolio_value'].iloc[0],
                label='Strategy', color='blue', linewidth=2)
        
        if benchmark_df is not None:
            ax1.plot(benchmark_df.index, 
                    benchmark_df['close'] / benchmark_df['close'].iloc[0],
                    label='Benchmark (Buy & Hold)', color='gray', 
                    linewidth=1.5, alpha=0.7)
        
        ax1.set_title('Cumulative Return', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Value')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 回撤图
        ax2 = axes[1]
        cummax = result_df['portfolio_value'].cummax()
        drawdown = (result_df['portfolio_value'] - cummax) / cummax * 100
        
        ax2.fill_between(result_df.index, drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown %')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        # 3. 每日收益分布
        ax3 = axes[2]
        returns = result_df['returns'].dropna() * 100
        ax3.hist(returns, bins=50, color='green', alpha=0.6, edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.2f}%')
        ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"回测图表已保存: {path}")
        else:
            default_path = Path('results/figures/backtest_result.png')
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            path = default_path
            logger.info(f"回测图表已保存: {default_path}")
        
        plt.close()
        return str(path)
    
    @staticmethod
    def plot_feature_importance(importance_dict: Dict[str, float],
                               save_path: Optional[str] = None,
                               top_n: int = 15) -> str:
        """
        绘制特征重要性图
        
        Args:
            importance_dict: 特征重要性字典
            save_path: 保存路径
            top_n: 显示前N个特征
        
        Returns:
            保存的文件路径
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib未安装")
            raise
        
        logger.info("生成特征重要性图...")
        
        # 排序并取前N
        sorted_items = sorted(importance_dict.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
        features = [item[0] for item in sorted_items]
        importance = [item[1] for item in sorted_items]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax.barh(range(len(features)), importance, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # 重要性高的在上面
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance (Top {})'.format(top_n), 
                    fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, importance)):
            ax.text(val + 0.001, i, f'{val:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            default_path = Path('results/figures/feature_importance.png')
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            path = default_path
        
        logger.info(f"特征重要性图已保存: {path}")
        plt.close()
        return str(path)
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, 
                               feature_cols: list,
                               save_path: Optional[str] = None) -> str:
        """
        绘制特征相关性热力图
        
        Args:
            df: 数据DataFrame
            feature_cols: 特征列名
            save_path: 保存路径
        
        Returns:
            保存的文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn未安装")
            raise
        
        logger.info("生成特征相关性图...")
        
        # 计算相关性
        corr_matrix = df[feature_cols].corr()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=0.5)
        
        ax.set_title('Feature Correlation Matrix', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            default_path = Path('results/figures/correlation_matrix.png')
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            path = default_path
        
        logger.info(f"相关性图已保存: {path}")
        plt.close()
        return str(path)
    
    @staticmethod
    def generate_report(result_df: pd.DataFrame, 
                       metrics: dict,
                       save_path: Optional[str] = None) -> str:
        """
        生成回测报告文本
        
        Args:
            result_df: 回测结果
            metrics: 指标字典
            save_path: 保存路径
        
        Returns:
            报告文本
        """
        from datetime import datetime
        
        report_lines = [
            "=" * 60,
            "Quant Stock Picker - Backtest Report",
            "=" * 60,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Performance Metrics:",
            "-" * 40,
            f"Total Return:       {metrics['total_return']:>10.2f}%",
            f"Annualized Return:  {metrics['annualized_return']:>10.2f}%",
            f"Volatility:         {metrics['volatility']:>10.2f}%",
            f"Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}",
            f"Max Drawdown:       {metrics['max_drawdown']:>10.2f}%",
            f"Win Rate:           {metrics['win_rate']:>10.2f}%",
            f"Profit/Loss Ratio:  {metrics['profit_loss_ratio']:>10.2f}",
            f"Total Trades:       {metrics['total_trades']:>10d}",
            f"Final Capital:      {metrics['final_capital']:>10,.2f}",
            "",
            "=" * 60,
        ]
        
        report_text = "\n".join(report_lines)
        
        # 保存
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(report_text)
            logger.info(f"报告已保存: {path}")
        
        return report_text
