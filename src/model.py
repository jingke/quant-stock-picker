"""
机器学习模型模块
支持XGBoost、Random Forest、LightGBM
"""

import logging
import pickle
from pathlib import Path
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class StockPredictor:
    """
    股票预测器
    
    支持多种模型:
    - XGBoost (默认，推荐)
    - Random Forest
    - LightGBM
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        初始化预测器
        
        Args:
            model_type: 模型类型 ('xgboost', 'random_forest', 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        
        logger.info(f"预测器初始化: model_type={model_type}")
    
    def build_model(self, **kwargs):
        """
        构建模型
        
        Args:
            **kwargs: 模型参数
        """
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                
                params = {
                    'n_estimators': kwargs.get('n_estimators', 100),
                    'max_depth': kwargs.get('max_depth', 5),
                    'learning_rate': kwargs.get('learning_rate', 0.1),
                    'subsample': kwargs.get('subsample', 0.8),
                    'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
                    'random_state': kwargs.get('random_state', 42),
                    'eval_metric': kwargs.get('eval_metric', 'logloss'),
                    'use_label_encoder': False
                }
                
                self.model = xgb.XGBClassifier(**params)
                logger.info("  ✓ XGBoost模型构建完成")
                
            except ImportError:
                logger.error("xgboost未安装，请运行: pip install xgboost")
                raise
        
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', 10),
                'random_state': kwargs.get('random_state', 42),
                'n_jobs': -1
            }
            
            self.model = RandomForestClassifier(**params)
            logger.info("  ✓ Random Forest模型构建完成")
        
        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                
                params = {
                    'n_estimators': kwargs.get('n_estimators', 100),
                    'max_depth': kwargs.get('max_depth', 5),
                    'learning_rate': kwargs.get('learning_rate', 0.1),
                    'random_state': kwargs.get('random_state', 42),
                    'verbose': -1
                }
                
                self.model = lgb.LGBMClassifier(**params)
                logger.info("  ✓ LightGBM模型构建完成")
                
            except ImportError:
                logger.error("lightgbm未安装，请运行: pip install lightgbm")
                raise
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        if self.model is None:
            raise ValueError("模型未构建，请先调用build_model()")
        
        logger.info(f"开始训练: {len(X_train)} 条样本")
        self.model.fit(X_train, y_train)
        logger.info("训练完成")
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        预测
        
        Args:
            X: 输入特征
        
        Returns:
            (预测类别, 预测概率)
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]  # 正类概率
        
        return y_pred, y_prob
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
        
        Returns:
            评估指标字典
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        y_pred, y_prob = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        logger.info("模型评估结果:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return metrics
    
    def feature_importance(self) -> dict:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典 {特征名: 重要性}
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            logger.warning("当前模型不支持特征重要性")
            return {}
        
        if len(self.feature_names) == len(importance):
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}
    
    def save(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和元信息
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"模型已保存: {filepath}")
    
    def load(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.model_type = save_data.get('model_type', 'unknown')
        self.feature_names = save_data.get('feature_names', [])
        
        logger.info(f"模型已加载: {filepath}")
        logger.info(f"  模型类型: {self.model_type}")
        logger.info(f"  特征数量: {len(self.feature_names)}")
