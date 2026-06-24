"""
超参数优化模块
支持Optuna自动调参

Author: jingke
Date: 2026-06-24
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("quant_stock_picker")


class HyperparameterOptimizer:
    """
    超参数优化器
    
    支持:
    - Optuna (推荐)
    - Grid Search
    - Random Search
    """
    
    def __init__(self, model_type: str = 'xgboost') -> None:
        """
        初始化优化器
        
        Args:
            model_type: 模型类型
        """
        self.model_type: str = model_type
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[Any] = None
        
        logger.info(f"超参数优化器初始化: model_type={model_type}")
    
    def optimize_optuna(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         n_trials: int = 50) -> Dict[str, Any]:
        """
        使用Optuna进行超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            n_trials: 优化次数
        
        Returns:
            最优参数字典
        """
        try:
            import optuna
            from sklearn.metrics import roc_auc_score
            
            logger.info(f"开始Optuna优化: n_trials={n_trials}")
            
            def objective(trial):
                if self.model_type == 'xgboost':
                    import xgboost as xgb
                    
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                        'random_state': 42,
                        'use_label_encoder': False,
                        'eval_metric': 'logloss'
                    }
                    
                    model = xgb.XGBClassifier(**params)
                
                elif self.model_type == 'random_forest':
                    from sklearn.ensemble import RandomForestClassifier
                    
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    
                    model = RandomForestClassifier(**params)
                
                elif self.model_type == 'lightgbm':
                    import lightgbm as lgb
                    
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': 42,
                        'verbose': -1
                    }
                    
                    model = lgb.LGBMClassifier(**params)
                
                else:
                    raise ValueError(f"不支持的模型类型: {self.model_type}")
                
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_val)
                
                return roc_auc_score(y_val, y_prob)
            
            # 创建study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            self.best_params = study.best_params
            self.study = study
            
            logger.info(f"Optuna优化完成:")
            logger.info(f"  最优AUC: {study.best_value:.4f}")
            logger.info(f"  最优参数: {study.best_params}")
            
            return study.best_params
            
        except ImportError:
            logger.error("optuna未安装，请运行: pip install optuna")
            raise
    
    def optimize_grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        使用Grid Search进行超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        
        Returns:
            最优参数字典
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score, make_scorer
        
        logger.info("开始Grid Search优化")
        
        if self.model_type == 'xgboost':
            import xgboost as xgb
            
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 合并训练集和验证集用于交叉验证
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        # 创建时间序列分割（避免数据泄露）
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring=make_scorer(roc_auc_score, needs_proba=True),
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        self.best_params = grid_search.best_params_
        
        logger.info(f"Grid Search完成:")
        logger.info(f"  最优AUC: {grid_search.best_score_:.4f}")
        logger.info(f"  最优参数: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        获取最优参数
        
        Returns:
            最优参数字典
        """
        return self.best_params
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径
        """
        if self.study is None:
            logger.warning("没有优化历史可绘制")
            return
        
        try:
            import optuna.visualization as vis
            import matplotlib.pyplot as plt
            
            fig = vis.plot_optimization_history(self.study)
            
            if save_path:
                fig.write_image(save_path)
                logger.info(f"优化历史图已保存: {save_path}")
            else:
                fig.show()
        
        except ImportError:
            logger.warning("optuna可视化需要plotly，请运行: pip install plotly")
