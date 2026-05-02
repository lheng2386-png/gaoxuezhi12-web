"""
高血脂风险分层预警与干预推荐系统 - 后端API入口
三大核心算法模块：
1. 高血脂关键风险因素识别 - Lasso-Logistic + XGBoost-SHAP + 随机森林交叉验证
2. 三级风险分层预警 - CART决策树 + 中西医融合三级标签
3. 个体化干预方案优化 - 画像分型 + 约束组合优化

Frontend → API → Backend → Models → Result → Frontend

Author: 基于论文CMC2604725研究成果
Version: 1.0
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os

from models.risk_classification import ThreeLevelRiskClassifier
from models.intervention_optimization import InterventionOptimizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)  # 支持跨域，前后端分离

# 初始化所有模型
# 这些模型在有训练数据时可以重新训练，默认使用论文提取的阈值规则
risk_classifier = ThreeLevelRiskClassifier()
intervention_optimizer = InterventionOptimizer()

print("=" * 60)
print("高血脂风险分层预警与干预推荐系统 - 后端API V1.0")
print("=" * 60)
print("✓ 关键风险因素识别模块: 按需加载 (Lasso-Logistic + XGBoost-SHAP + RF)")
print("✓ 三级风险分层预警模块: 已加载 (CART决策树，论文准确率95.00%)")
print("✓ 个体化干预优化模块: 已加载 (画像分型 + 约束组合优化)")
print("✓ CORS已启用，支持前后端分离")
print()
print("API端点:")
print("  GET  /                 - Web应用首页")
print("  POST /api/predict       - 完整风险评估和干预推荐")
print("  POST /api/identify      - 关键风险因素识别")
print("  GET  /api/thresholds    - 获取论文提取的阈值信息")
print("=" * 60)


@app.route('/')
def web_index():
    """Web应用首页"""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/about')
def web_about():
    """关于页面"""
    return send_from_directory(FRONTEND_DIR, 'about.html')


@app.route('/api')
def api_index():
    """API首页"""
    return jsonify({
        'name': '高血脂风险分层预警与干预推荐系统',
        'version': '1.0',
        'description': '基于中西医融合信息，实现风险识别、分层预警、干预推荐一体化',
        'modules': [
            '关键风险因素识别 (Lasso-Logistic + XGBoost-SHAP + 随机森林)',
            '三级风险分层预警 (CART决策树)',
            '个体化干预方案优化 (画像分型 + 约束组合优化)'
        ],
        'paper_accuracy': 0.95,
        'paper_source': 'CMC2604725'
    })


@app.route('/api/thresholds', methods=['GET'])
def get_thresholds():
    """获取论文提取的阈值信息"""
    return jsonify({
        'code': 0,
        'data': risk_classifier.thresholds
    })


@app.route('/api/identify', methods=['POST'])
def identify_factors():
    """
    关键风险因素识别API
    需要传入:
    {
        "data": [
            {"feature1": value1, ..., "label": 0/1},
            ...
        ]
    }
    """
    try:
        from models.feature_identification import KeyRiskFactorIdentifier
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({
                'code': 1,
                'message': '缺少data字段',
                'data': None
            })
        
        # 转换为DataFrame
        import pandas as pd
        df = pd.DataFrame(data['data'])
        X = df.drop(columns=['label'])
        y = df['label']

        feature_identifier = KeyRiskFactorIdentifier()
        
        # 执行完整识别流程
        result = feature_identifier.fit_all(X, y)
        
        # 获取权重汇总
        summary_df = feature_identifier.get_feature_weight_summary(result)
        
        return jsonify({
            'code': 0,
            'message': 'success',
            'data': {
                'consensus_features': result['consensus_features'],
                'n_consensus': result['n_consensus'],
                'paper_core_found': result['paper_core_found'],
                'lasso_result': {
                    'selected_features': result['lasso_result']['selected_features'],
                    'feature_coef': result['lasso_result']['feature_coef']
                },
                'xgb_result': {
                    'feature_importance': result['xgb_result']['feature_importance']
                },
                'rf_result': {
                    'mean_cv_accuracy': result['rf_result']['cv_mean_accuracy'],
                    'feature_importance': result['rf_result']['feature_importance']
                },
                'summary': summary_df.to_dict(orient='records')
            }
        })
    except Exception as e:
        return jsonify({
            'code': 2,
            'message': str(e),
            'data': None
        })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    完整风险评估和干预推荐API
    输入:
    {
        "age": 55,
        "gender": 1,
        "height": 170,
        "weight": 75,
        "tan_score": 45,
        "tg": 1.5,
        "tc": 5.2,
        "hdl_c": 1.2,
        "ldl_c": 3.4,
        "urea_acid": 360,
        "adl_score": 55,
        "budget": 2000
    }
    """
    try:
        data = request.get_json() or {}

        required_fields = [
            'age', 'gender', 'height', 'weight', 'tan_score',
            'tg', 'tc', 'hdl_c', 'ldl_c', 'urea_acid', 'adl_score'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'code': 1,
                'message': f"缺少必填字段: {', '.join(missing_fields)}",
                'data': None
            })
        
        # 计算BMI
        height_m = data['height'] / 100
        bmi = data['weight'] / (height_m ** 2)
        
        # 整理特征（对齐模型期望的特征名）
        features = {
            'age': data['age'],
            'gender': data['gender'],
            'bmi': bmi,
            'tan_score_raw': data['tan_score'],
            'tg': data['tg'],
            'tc': data['tc'],
            'hdl_c': data['hdl_c'],
            'ldl_c': data['ldl_c'],
            'urea_acid': data['urea_acid'],
            'activity_score': data['adl_score']
        }
        
        # 第一步: 风险分层预测
        risk_result = risk_classifier.predict_single(features)
        
        # 第二步: 患者画像分型
        patient_type_result = intervention_optimizer.classify_patient(features)
        
        # 第三步: 干预优化
        budget = data.get('budget', 2000)
        intervention_result = intervention_optimizer.optimize(
            patient_type_result['patient_type'],
            max_budget=budget,
            features=features
        )
        
        # 获取多个备选方案
        alternative_plans = intervention_optimizer.get_top_n_plans(
            patient_type_result['patient_type'],
            max_budget=budget,
            features=features,
            n=3
        )
        intervention_result['alternatives'] = alternative_plans
        
        return jsonify({
            'code': 0,
            'message': 'success',
            'data': {
                'input': {
                    'age': data['age'],
                    'bmi': round(bmi, 1),
                    'tan_score': data['tan_score'],
                    'tg': data['tg'],
                    'tc': data['tc'],
                    'ldl_c': data['ldl_c'],
                    'hdl_c': data['hdl_c'],
                    'urea_acid': data['urea_acid'],
                    'activity_score': data['adl_score'],
                    'budget': budget
                },
                'risk': risk_result,
                'patient_type': patient_type_result,
                'intervention': intervention_result,
                'alternatives': alternative_plans,
                'paper_accuracy': risk_classifier.config['model_accuracy']['cart_accuracy']
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'code': 1,
            'message': str(e),
            'data': None
        })


@app.route('/api/alternatives', methods=['POST'])
def get_alternatives():
    """获取多个备选干预方案"""
    try:
        data = request.get_json()
        patient_type = data.get('patient_type', 'metabolic')
        max_budget = data.get('max_budget', 2000)
        n = data.get('n', 3)
        
        plans = intervention_optimizer.get_top_n_plans(patient_type, max_budget, n)
        
        return jsonify({
            'code': 0,
            'message': 'success',
            'data': {
                'plans': plans,
                'count': len(plans)
            }
        })
    except Exception as e:
        return jsonify({
            'code': 1,
            'message': str(e),
            'data': None
        })


if __name__ == '__main__':
    # 启动服务
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
