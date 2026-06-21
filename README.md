# Quant Stock Picker: Intelligent A-Share Selection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 项目简介

基于机器学习的 A 股智能选股系统，通过融合技术指标、财务因子和宏观数据，构建多因子选股模型，并进行完整回测验证。

**核心特点**：
- 多因子融合：技术面 + 基本面 + 宏观面
- 模型对比：XGBoost vs Random Forest vs LightGBM
- 完整回测：夏普比率、最大回撤、胜率分析
- 可解释性：特征重要性分析

## 技术栈

- **数据处理**: pandas, numpy, akshare, yfinance
- **技术指标**: talib
- **机器学习**: scikit-learn, xgboost, lightgbm
- **可视化**: matplotlib, seaborn
- **回测**: 自研回测框架

## 项目结构

```
quant-stock-picker/
├── main.py              # 主入口
├── config/
│   └── config.yaml      # 配置文件
├── src/                 # 核心代码
│   ├── data_fetcher.py      # 数据获取
│   ├── feature_engineer.py # 特征工程
│   ├── model.py             # 机器学习模型
│   ├── backtester.py        # 回测框架
│   └── evaluator.py         # 可视化评估
├── tests/               # 单元测试
│   └── test_modules.py
├── notebooks/           # 探索性分析
├── data/                # 数据目录
│   ├── raw/
│   └── processed/
├── results/             # 回测结果
│   └── figures/
├── models/              # 保存的模型
├── requirements.txt     # 依赖
└── README.md           # 本文档
```

## 快速开始

### 演示运行（无需安装数据源）

```bash
# 直接运行演示脚本，生成示例回测结果和图表
python demo.py
```

输出文件：
- `results/figures/backtest_demo.png` - 回测可视化
- `results/figures/feature_importance_demo.png` - 特征重要性
- `results/demo_report.txt` - 性能报告

### 完整系统运行

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/jingke/quant-stock-picker.git
cd quant-stock-picker

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
python main.py --config config/config.yaml
```

### 3. 分步运行

```bash
# 仅获取数据
python main.py --mode data --config config/config.yaml

# 仅训练模型
python main.py --mode train --config config/config.yaml

# 仅回测
python main.py --mode backtest --config config/config.yaml

# 预测单只股票
python main.py --mode predict --symbol 000001 --config config/config.yaml
```

## 核心结果

| 指标 | 策略 | 买入持有 |
|------|------|---------|
| 年化收益 | 100.83% | 8.5% |
| 夏普比率 | 4.29 | 0.68 |
| 最大回撤 | -12.14% | -25.6% |
| 胜率 | 33.79% | - |

> 注：以上为模拟数据演示结果，实际表现取决于市场环境

### 回测可视化

![回测结果](results/figures/backtest_demo.png)

*累计收益曲线 vs 回撤分析 vs 日收益分布*

### 特征重要性

![特征重要性](results/figures/feature_importance_demo.png)

*Top 10 关键特征：MA均线、成交量比、价格波动等*

## 方法论

### 1. 特征工程

- **技术指标**：MA、RSI、MACD、布林带、KDJ
- **财务因子**：PE、PB、ROE、营收增长率
- **宏观因子**：利率、CPI、PMI

### 2. 模型训练

- 分类目标：未来5日是否上涨
- 训练集：2018-2021
- 验证集：2022
- 测试集：2023

### 3. 回测规则

- 初始资金：10万
- 交易费用：0.05%（单边）
- 滑点：0.1%
- 信号阈值：0.6（概率 > 0.6 则买入）

## 配置文件

编辑 `config/config.yaml` 自定义参数：

```yaml
data:
  market: "A_share"           # A股/美股/港股
  symbols: ["000001", "000002"]  # 股票列表

model:
  type: "xgboost"           # 模型类型
  prediction_horizon: 5     # 预测周期（天）

backtest:
  initial_capital: 100000    # 初始资金
  signal_threshold: 0.6     # 买入阈值
```

## 测试

```bash
pytest tests/ -v
```

## 未来改进

- [ ] 引入 LSTM/Transformer 时序模型
- [ ] 多因子组合优化（马科维茨）
- [ ] 实时数据接入
- [ ] 行业轮动策略
- [ ] 多股票组合回测

## 关于我

利兹大学计算机科学本科，对量化金融和机器学习交叉领域感兴趣。

**本项目用途**：作为 FinTech / Data Science 研究生申请的作品集项目

联系：your.email@example.com

## License

MIT License
