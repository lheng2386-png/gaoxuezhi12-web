from __future__ import annotations

"""
模块二：三级风险分层预警
算法：CART决策树、中西医融合三级标签（低/中/高风险）
论文对应：问题二 - 基于临床先验与数据校准的三级风险预警模型

论文结论:
- 痰湿积分约 60 分、TG 约 1.7 mmol/L 、TC 约 6.2 mmol/L 及活动能力约 40 分是风险分层的重要界值
- CART 模型 5 折交叉验证准确率达到 95.00%
- 高风险人群主要表现为"痰湿偏颇 + 血脂异常 + 活动受限"的复合画像

Author: 基于论文CMC2604725研究成果
"""
import json
import os
try:
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError:
    np = None
    pd = None
    DecisionTreeClassifier = None
    export_text = None
    cross_val_score = None
    StratifiedKFold = None

    class BaseEstimator:
        pass

    class ClassifierMixin:
        pass
from typing import Dict, List, Tuple, Optional, Any


class ThreeLevelRiskClassifier(BaseEstimator, ClassifierMixin):
    """
    三级风险分层分类器
    融合论文提取的先验阈值知识和CART决策树学习
    """
    
    def __init__(
        self,
        config_path: str = None,
        max_depth: int = 5,
        min_samples_split: int = 20,
        random_state: int = 42
    ):
        if config_path is None:
            # 默认配置路径（相对于当前文件）
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, '../config/risk_thresholds.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.thresholds = self.config['thresholds']
        self.risk_definitions = self.config['risk_level_definition']
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        # 初始化CART模型。Web预测主流程可直接使用论文阈值规则，不强制安装scikit-learn。
        self.model = None
        if DecisionTreeClassifier is not None:
            self.model = DecisionTreeClassifier(
                criterion='gini',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
        
        self.is_fitted = False
        self.feature_names_ = None
    
    def get_prior_label(
        self,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        基于论文提取的先验阈值生成三级标签
        
        参数:
            features: 特征DataFrame，必须包含所需字段
            
        返回:
            labels: 0=低风险, 1=中风险, 2=高风险
        """
        if np is None:
            raise ImportError("get_prior_label 需要安装 numpy/pandas")

        n_samples = len(features)
        risk_score = np.zeros(n_samples)
        
        # 痰湿积分检查（分数越高风险越高）
        if 'tan_score_raw' in features.columns:
            tan_score = features['tan_score_raw'].values
            high_risk_cutoff = self.thresholds['tan_score']['high_risk_cutoff']
            medium_risk_cutoff = self.thresholds['tan_score']['medium_risk_cutoff']
            risk_score += np.where(tan_score >= high_risk_cutoff, 2, 
                                 np.where(tan_score >= medium_risk_cutoff, 1, 0))
        
        # TG检查
        if 'tg' in features.columns:
            tg = features['tg'].values
            high_risk_cutoff = self.thresholds['tg']['high_risk_cutoff']
            medium_risk_cutoff = self.thresholds['tg']['medium_risk_cutoff']
            risk_score += np.where(tg >= high_risk_cutoff, 2, 
                                 np.where(tg >= medium_risk_cutoff, 1, 0))
        
        # TC检查
        if 'tc' in features.columns:
            tc = features['tc'].values
            high_risk_cutoff = self.thresholds['tc']['high_risk_cutoff']
            medium_risk_cutoff = self.thresholds['tc']['medium_risk_cutoff']
            risk_score += np.where(tc >= high_risk_cutoff, 2, 
                                 np.where(tc >= medium_risk_cutoff, 1, 0))
        
        # 活动能力检查（分数越低风险越高）
        if 'activity_score' in features.columns:
            activity = features['activity_score'].values
            high_risk_cutoff = self.thresholds['activity_score']['high_risk_cutoff']
            risk_score += np.where(activity <= high_risk_cutoff, 2, 0)
        
        # LDL-C检查
        if 'ldl_c' in features.columns and 'high_risk_cutoff' in self.thresholds['ldl_c']:
            ldl = features['ldl_c'].values
            high_risk_cutoff = self.thresholds['ldl_c']['high_risk_cutoff']
            medium_risk_cutoff = self.thresholds['ldl_c']['medium_risk_cutoff']
            risk_score += np.where(ldl >= high_risk_cutoff, 2, 
                                 np.where(ldl >= medium_risk_cutoff, 1, 0))
        
        # HDL-C检查（越低风险越高）
        if 'hdl_c' in features.columns and 'low_risk_cutoff' in self.thresholds['hdl_c']:
            hdl = features['hdl_c'].values
            low_risk_cutoff = self.thresholds['hdl_c']['low_risk_cutoff']
            risk_score += np.where(hdl <= low_risk_cutoff, 2, 0)
        
        # BMI检查
        if 'bmi' in features.columns:
            bmi = features['bmi'].values
            high_risk_cutoff = self.thresholds['bmi']['high_risk_cutoff']
            medium_risk_cutoff = self.thresholds['bmi']['medium_risk_cutoff']
            risk_score += np.where(bmi >= high_risk_cutoff, 2, 
                                 np.where(bmi >= medium_risk_cutoff, 1, 0))
        
        # 血尿酸检查
        if 'urea_acid' in features.columns:
            ua = features['urea_acid'].values
            high_risk_cutoff = self.thresholds['urea_acid']['high_risk_cutoff']
            medium_risk_cutoff = self.thresholds['urea_acid']['medium_risk_cutoff']
            risk_score += np.where(ua >= high_risk_cutoff, 2, 
                                 np.where(ua >= medium_risk_cutoff, 1, 0))
        
        # 根据总分转换为三级标签
        # 0-2: 低风险, 3-5: 中风险, 6+: 高风险
        labels = np.where(risk_score <= 2, 0,
                 np.where(risk_score <= 5, 1, 2))
        
        return labels
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None
    ) -> 'ThreeLevelRiskClassifier':
        """
        训练CART决策树模型
        如果y为None，则使用先验阈值生成标签
        
        参数:
            X: 特征矩阵
            y: 可选，真实标签，如果为None则使用先验生成
            
        返回:
            self
        """
        if self.model is None:
            raise ImportError("训练CART模型需要安装 scikit-learn")

        self.feature_names_ = X.columns.tolist()
        
        # 如果没有提供标签，使用先验阈值生成
        if y is None:
            y = self.get_prior_label(X)
        
        # 训练CART模型
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"CART决策树训练完成")
        print(f"  输入特征: {len(self.feature_names_)}")
        print(f"  树深度: {self.model.get_depth()}")
        print(f"  叶子节点数: {self.model.get_n_leaves()}")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        预测风险等级
        
        参数:
            X: 特征矩阵
            
        返回:
            预测标签: 0=低, 1=中, 2=高
        """
        if not self.is_fitted:
            # 如果未训练，直接使用规则预测
            return self.get_prior_label(X)
        
        return self.model.predict(X)
    
    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        return self.model.predict_proba(X)
    
    def cross_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        交叉验证评估模型性能
        
        参数:
            X: 特征矩阵
            y: 真实标签
            cv_folds: 交叉验证折数
            
        返回:
            评估结果字典
        """
        if StratifiedKFold is None or cross_val_score is None:
            raise ImportError("交叉验证需要安装 scikit-learn")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
        
        result = {
            'cv_folds': cv_folds,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
        
        print(f"{cv_folds}折交叉验证结果:")
        print(f"  平均准确率: {scores.mean():.2%} (±{scores.std():.2%})")
        
        # 论文报告准确率为 95.00%，比较是否一致
        paper_accuracy = self.config['model_accuracy']['cart_accuracy']
        print(f"  论文报告准确率: {paper_accuracy:.2%}")
        
        return result
    
    def extract_rules(self) -> str:
        """提取决策树规则文本"""
        if not self.is_fitted:
            return "模型尚未训练"
        
        if self.feature_names_ is None:
            return "特征名称未记录"
        
        rules = export_text(self.model, feature_names=self.feature_names_)
        return rules
    
    def predict_single(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        单样本预测，返回详细结果
        
        参数:
            features: 特征字典
            
        返回:
            预测结果字典，包含风险等级、异常因素等
        """
        # 转换为DataFrame
        X = pd.DataFrame([features]) if pd is not None else None
        
        # 基于规则识别异常因素
        risk_factors = []
        risk_level = 0
        
        # 逐一检查
        # 痰湿积分
        tan_score = features.get('tan_score_raw', 0)
        if tan_score >= self.thresholds['tan_score']['high_risk_cutoff']:
            risk_factors.append(f"痰湿体质积分过高({tan_score:.0f}分)")
        elif tan_score >= self.thresholds['tan_score']['medium_risk_cutoff']:
            risk_factors.append(f"痰湿体质积分偏高({tan_score:.0f}分)")
        
        # TG
        tg = features.get('tg', 0)
        if tg >= self.thresholds['tg']['high_risk_cutoff']:
            risk_factors.append(f"甘油三酯过高({tg:.1f} mmol/L)")
        elif tg >= self.thresholds['tg']['medium_risk_cutoff']:
            risk_factors.append(f"甘油三酯偏高({tg:.1f} mmol/L)")
        
        # TC
        tc = features.get('tc', 0)
        if tc >= self.thresholds['tc']['high_risk_cutoff']:
            risk_factors.append(f"总胆固醇过高({tc:.1f} mmol/L)")
        elif tc >= self.thresholds['tc']['medium_risk_cutoff']:
            risk_factors.append(f"总胆固醇偏高({tc:.1f} mmol/L)")
        
        # 活动能力
        activity = features.get('activity_score', 100)
        if activity <= self.thresholds['activity_score']['high_risk_cutoff']:
            risk_factors.append(f"活动能力明显受限({activity:.0f}分)")
        
        # LDL-C
        ldl_c = features.get('ldl_c', 0)
        if 'high_risk_cutoff' in self.thresholds['ldl_c'] and ldl_c >= self.thresholds['ldl_c']['high_risk_cutoff']:
            risk_factors.append(f"低密度脂蛋白胆固醇过高({ldl_c:.1f} mmol/L)")
        elif 'medium_risk_cutoff' in self.thresholds['ldl_c'] and ldl_c >= self.thresholds['ldl_c']['medium_risk_cutoff']:
            risk_factors.append(f"低密度脂蛋白胆固醇偏高({ldl_c:.1f} mmol/L)")
        
        # HDL-C
        hdl_c = features.get('hdl_c', 3)
        if hdl_c <= self.thresholds['hdl_c']['low_risk_cutoff']:
            risk_factors.append(f"高密度脂蛋白胆固醇偏低({hdl_c:.1f} mmol/L)")
        
        # BMI
        bmi = features.get('bmi', 22)
        if bmi >= self.thresholds['bmi']['high_risk_cutoff']:
            risk_factors.append(f"BMI超标(肥胖，{bmi:.1f} kg/m²)")
        elif bmi >= self.thresholds['bmi']['medium_risk_cutoff']:
            risk_factors.append(f"BMI超重({bmi:.1f} kg/m²)")
        
        # 血尿酸
        urea_acid = features.get('urea_acid', 300)
        if urea_acid >= self.thresholds['urea_acid']['high_risk_cutoff']:
            risk_factors.append(f"血尿酸过高({urea_acid:.0f} μmol/L)")
        elif urea_acid >= self.thresholds['urea_acid']['medium_risk_cutoff']:
            risk_factors.append(f"血尿酸偏高({urea_acid:.0f} μmol/L)")
        
        # 使用模型预测
        if self.is_fitted and X is not None:
            predicted_level = int(self.predict(X)[0])
        else:
            # 使用规则分级
            total_score = len([f for f in risk_factors if '过高' in f or '超标' in f]) * 2 + \
                         len([f for f in risk_factors if '偏高' in f or '超重' in f]) * 1
            if total_score <= 2:
                predicted_level = 0
            elif total_score <= 5:
                predicted_level = 1
            else:
                predicted_level = 2
        
        level_map = {0: 'low', 1: 'medium', 2: 'high'}
        risk_level_key = level_map[predicted_level]
        risk_info = self.risk_definitions[risk_level_key]
        
        result = {
            'risk_level_code': predicted_level,
            'risk_level': risk_level_key,
            'risk_level_name': risk_info['name'],
            'risk_color': risk_info['color'],
            'description': risk_info['description'],
            'risk_factors': risk_factors,
            'number_of_risk_factors': len(risk_factors),
            'paper_thresholds_used': {
                'tan_score_cutoff': self.thresholds['tan_score']['high_risk_cutoff'],
                'tg_cutoff': self.thresholds['tg']['high_risk_cutoff'],
                'tc_cutoff': self.thresholds['tc']['high_risk_cutoff'],
                'activity_cutoff': self.thresholds['activity_score']['high_risk_cutoff']
            }
        }
        
        return result
    
    def get_risk_definition(self, risk_level: str) -> Dict:
        """获取风险等级定义"""
        return self.risk_definitions[risk_level]


# 配置文件 - 使用相对于backend目录的路径
import os
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# 复制配置文件
config_path = os.path.join(config_dir, 'risk_thresholds.json')
if not os.path.exists(config_path):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
        "name": "高血脂风险分层阈值配置（来自论文研究结果）",
        "description": "基于CART决策树提取的阈值边界",
        "version": "1.0",
        "thresholds": {
            "tan_score": {
                "high_risk_cutoff": 60,
                "medium_risk_cutoff": 40,
                "unit": "分",
                "description": "痰湿体质积分"
            },
            "tg": {
                "high_risk_cutoff": 1.7,
                "medium_risk_cutoff": 1.2,
                "unit": "mmol/L",
                "description": "甘油三酯"
            },
            "tc": {
                "high_risk_cutoff": 6.2,
                "medium_risk_cutoff": 5.2,
                "unit": "mmol/L",
                "description": "总胆固醇"
            },
            "activity_score": {
                "high_risk_cutoff": 40,
                "unit": "分",
                "description": "活动能力评分，分数越低风险越高",
                "reverse": True
            },
            "ldl_c": {
                "high_risk_cutoff": 4.1,
                "medium_risk_cutoff": 3.4,
                "unit": "mmol/L",
                "description": "低密度脂蛋白胆固醇"
            },
            "hdl_c": {
                "low_risk_cutoff": 1,
                "unit": "mmol/L",
                "description": "高密度脂蛋白胆固醇，越低风险越高",
                "reverse": True
            },
            "urea_acid": {
                "high_risk_cutoff": 420,
                "medium_risk_cutoff": 360,
                "unit": "μmol/L",
                "description": "血尿酸"
            },
            "bmi": {
                "high_risk_cutoff": 28,
                "medium_risk_cutoff": 24,
                "unit": "kg/m²",
                "description": "体重指数"
            }
        },
        "risk_level_definition": {
            "low": {
                "name": "低风险",
                "color": "#28a745",
                "description": "各项指标基本正常，保持健康生活方式即可"
            },
            "medium": {
                "name": "中风险",
                "color": "#ffc107",
                "description": "部分指标异常，需要关注并适当调整生活方式"
            },
            "high": {
                "name": "高风险",
                "color": "#dc3545",
                "description": "多项指标异常，建议就医并进行针对性干预"
            }
        },
        "model_accuracy": {
            "cart_accuracy": 0.95,
            "cv_folds": 5,
            "source": "论文交叉验证结果"
        }
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 测试
    print("=== 三级风险分层预警 - 测试 ===")
    
    classifier = ThreeLevelRiskClassifier()
    
    # 测试一个高风险案例
    test_high_risk = {
        'tan_score_raw': 65,
        'tg': 2.1,
        'tc': 6.8,
        'hdl_c': 0.9,
        'ldl_c': 4.5,
        'urea_acid': 450,
        'bmi': 29,
        'activity_score': 35
    }
    
    result = classifier.predict_single(test_high_risk)
    print("\n高风险案例测试:")
    print(f"风险等级: {result['risk_level_name']}")
    print(f"风险因素: {result['risk_factors']}")
    print()
    
    # 测试一个低风险案例
    test_low_risk = {
        'tan_score_raw': 30,
        'tg': 1.0,
        'tc': 4.5,
        'hdl_c': 1.5,
        'ldl_c': 2.8,
        'urea_acid': 320,
        'bmi': 22,
        'activity_score': 60
    }
    
    result = classifier.predict_single(test_low_risk)
    print("低风险案例测试:")
    print(f"风险等级: {result['risk_level_name']}")
    print(f"风险因素: {result['risk_factors']}")
