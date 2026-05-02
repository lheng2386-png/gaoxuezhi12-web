"""
模块三：个体化干预方案优化
算法：画像分型（代谢异常/肥胖并发/功能受限）+ 约束组合优化
论文对应：问题三 - 画像分型→约束筛选→组合优化→收益评估

论文结论:
- 代谢异常主导型患者更适合优先强化调理
- 肥胖并发型患者更适合采用"调理 + 运动"联合干预
- 功能受限型患者则更适合低强度、稳步改善方案
- 约束条件：6个月总成本不超过2000元

Author: 基于论文CMC2604725研究成果
"""
import json
import os
from typing import Dict, List, Tuple, Optional, Any


class InterventionOptimizer:
    """
    个体化干预方案优化器
    在给定约束（类型约束、预算约束）下最大化预期改善效果
    """
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, '../config/intervention_costs.json')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.treatment_levels = self.config['调理等级']
        self.exercise_plans = self.config['运动方案']
        self.default_constraints = self.config['default_constraints']
        self.type_rules = self.config['patient_type_rules']
        
        # 论文中的患者分型定义
        self.patient_type_definitions = {
            'metabolic': {
                'name': '代谢异常主导型',
                'description': '血脂异常为主，体质和活动能力尚可',
                'prefer_strategy': '优先强化调理'
            },
            'obesity': {
                'name': '肥胖并发型',
                'description': 'BMI超标 + 痰湿质 + 代谢异常',
                'prefer_strategy': '调理 + 运动联合干预'
            },
            'function_limited': {
                'name': '功能受限型',
                'description': '活动能力下降 + 高龄 + 多种并发症',
                'prefer_strategy': '低强度稳步改善方案'
            }
        }
    
    def classify_patient(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        根据用户特征进行患者画像分型
        基于论文中的分型规则
        
        参数:
            features: 特征字典，包含 age, bmi, tan_score_raw, activity_score 等
            
        返回:
            分型结果字典
        """
        age = features.get('age', 50)
        bmi = features.get('bmi', 24)
        tan_score = features.get('tan_score_raw', 30)
        activity = features.get('activity_score', 60)
        
        # 论文中的判断逻辑
        # 优先判断功能受限型
        if activity <= 40 or age >= 75:
            patient_type = 'function_limited'
        # 然后判断肥胖并发型
        elif bmi >= 28 and tan_score >= 40:
            patient_type = 'obesity'
        # 默认代谢异常主导型
        elif bmi < 28 and activity > 40:
            patient_type = 'metabolic'
        # 其他情况根据特征判断
        else:
            if bmi >= 24:
                patient_type = 'obesity'
            elif age >= 65:
                patient_type = 'function_limited'
            else:
                patient_type = 'metabolic'
        
        type_info = self.patient_type_definitions[patient_type]
        
        result = {
            'patient_type': patient_type,
            'type_name': type_info['name'],
            'description': type_info['description'],
            'recommended_strategy': type_info['prefer_strategy'],
            'paper_conclusion': self._get_paper_conclusion(patient_type)
        }
        
        return result
    
    def _get_paper_conclusion(self, patient_type: str) -> str:
        """获取论文对该类型的结论"""
        conclusions = {
            'metabolic': '代谢异常主导型患者更适合优先强化调理',
            'obesity': '肥胖并发型患者更适合采用"调理 + 运动"联合干预',
            'function_limited': '功能受限型患者则更适合低强度、稳步改善方案'
        }
        return conclusions.get(patient_type, '')
    
    def _get_constraints_for_type(self, patient_type: str) -> Dict[str, Any]:
        """获取对应类型的约束条件"""
        if patient_type in self.type_rules:
            return self.type_rules[patient_type]['constraints']
        return {}
    
    def generate_all_combinations(
        self,
        patient_type: str,
        max_budget: int,
        features: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        根据约束生成所有可行的干预组合
        
        参数:
            patient_type: 患者类型
            max_budget: 最大预算（6个月）
            
        返回:
            可行组合列表
        """
        constraints = self._get_constraints_for_type(patient_type)
        features = features or {}
        combinations = []
        
        # 调理等级范围
        tan_score = features.get('tan_score_raw')
        if tan_score is not None:
            if tan_score <= 58:
                min_treat = max_treat = 1
            elif tan_score <= 61:
                min_treat = max_treat = 2
            else:
                min_treat = max_treat = 3
        else:
            min_treat = constraints.get('min_treatment_level', 1)
            max_treat = constraints.get('max_treatment_level', 3)
        treat_keys = [f'level_{i}' for i in range(min_treat, max_treat + 1)]
        
        allowed_intensities = self._allowed_intensities(features, constraints)
        
        # 分型策略的最低强度要求过滤
        min_intensity = constraints.get('min_exercise_intensity', None)
        if min_intensity == 'medium':
            allowed_intensities = [i for i in allowed_intensities if i != 'intensity_low']
        elif min_intensity == 'high':
            allowed_intensities = ['intensity_high']
        
        # 遍历所有调理等级
        for treat_key in treat_keys:
            treat = self.treatment_levels[treat_key]
            
            # 不运动的情况
            if not constraints.get('require_exercise', False):
                total_cost = treat['cost_per_month'] * 6
                if total_cost <= max_budget:
                    total_effect = treat.get('effect_per_month_percent', 0) * 6
                    combinations.append({
                        'treatment': treat,
                        'treatment_key': treat_key,
                        'treatment_level': int(treat_key.split('_')[1]),
                        'treatment_cost_6months': total_cost,
                        'exercise': None,
                        'exercise_key': None,
                        'intensity_key': None,
                        'frequency': None,
                        'total_cost_6months': total_cost,
                        'total_effect_6months': total_effect,
                        'expected_tan_score_drop_percent': total_effect
                    })
            
            # 带运动的组合：原题频率为每周 1-10 次，6个月按24周计算
            for intensity_key in allowed_intensities:
                intensity = self.exercise_plans[intensity_key]
                intensity_level = intensity['level']
                max_frequency = constraints.get('max_frequency_per_week', 10)
                for frequency_per_week in range(1, max_frequency + 1):
                    exercise_cost = intensity['cost_per_session'] * frequency_per_week * 24
                    treatment_cost = treat['cost_per_month'] * 6
                    total_cost = treatment_cost + exercise_cost
                    if total_cost <= max_budget:
                        exercise_effect = self._exercise_effect_percent(intensity_level, frequency_per_week)
                        treatment_effect = treat.get('effect_per_month_percent', 0) * 6
                        total_effect = treatment_effect + exercise_effect
                        exercise = {
                            'duration_minutes': intensity['duration_minutes'],
                            'cost_per_session': intensity['cost_per_session'],
                            'frequency_per_week': frequency_per_week,
                            'cost_6months': exercise_cost,
                            'effect_percent_6months': exercise_effect,
                            'tolerance': intensity['tolerance']
                        }
                        combinations.append({
                            'treatment': treat,
                            'treatment_key': treat_key,
                            'treatment_level': int(treat_key.split('_')[1]),
                            'treatment_cost_6months': treatment_cost,
                            'exercise': exercise,
                            'exercise_key': f'frequency_{frequency_per_week}x',
                            'intensity_key': intensity_key,
                            'frequency_name': f'每周{frequency_per_week}次',
                            'frequency_per_week': frequency_per_week,
                            'intensity_name': intensity['name'],
                            'intensity_level': intensity_level,
                            'duration_minutes': intensity['duration_minutes'],
                            'total_cost_6months': total_cost,
                            'total_effect_6months': total_effect,
                            'expected_tan_score_drop_percent': total_effect
                        })
        
        return combinations

    def _allowed_intensities(self, features: Dict[str, float], constraints: Dict[str, Any]) -> List[str]:
        """根据原题年龄约束和活动量表总分约束确定可选运动强度"""
        intensity_map = {
            'low': ['intensity_low'],
            'medium': ['intensity_low', 'intensity_medium'],
            'high': ['intensity_low', 'intensity_medium', 'intensity_high']
        }
        age = features.get('age')
        activity = features.get('activity_score')

        max_level = 3
        if age is not None:
            if age >= 80:
                max_level = min(max_level, 1)
            elif age >= 60:
                max_level = min(max_level, 2)

        if activity is not None:
            if activity < 40:
                max_level = min(max_level, 1)
            elif activity < 60:
                max_level = min(max_level, 2)

        max_intensity = constraints.get('max_exercise_intensity', 'high')
        allowed = intensity_map.get(max_intensity, intensity_map['high'])
        level_map = {'intensity_low': 1, 'intensity_medium': 2, 'intensity_high': 3}
        return [item for item in allowed if level_map[item] <= max_level]

    def _exercise_effect_percent(self, intensity_level: int, frequency_per_week: int) -> float:
        """
        原题假设：每周训练少于5次时痰湿积分基本稳定；每周5次时，
        每提升一级强度，每月预期下降3%；同强度下每周增加1次，每月多下降1%。
        """
        if frequency_per_week < 5:
            return 0
        monthly_percent = intensity_level * 3 + max(0, frequency_per_week - 5)
        return monthly_percent * 6
    
    def optimize(
        self,
        patient_type: str,
        max_budget: Optional[int] = None,
        features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        寻找最优干预方案
        
        参数:
            patient_type: 患者类型
            max_budget: 最大预算（6个月），默认2000元
            
        返回:
            最优方案结果
        """
        if max_budget is None:
            max_budget = self.default_constraints['max_total_cost_6months']
        
        type_info = self.patient_type_definitions[patient_type]
        all_combinations = self.generate_all_combinations(patient_type, max_budget, features)
        
        if not all_combinations:
            # 如果没有满足约束的组合，放宽约束重试
            print(f"警告：没有找到满足当前约束的组合，尝试放宽约束")
            # 放宽一级调理等级
            all_combinations = self.generate_all_combinations(patient_type, max_budget + 500, features)
        
        # 按效果降序排序
        all_combinations.sort(key=lambda x: x['total_effect_6months'], reverse=True)
        
        # 帕累托优化：如果效果相近但成本更低，选择成本更低的
        best = all_combinations[0] if all_combinations else None
        if all_combinations and len(all_combinations) > 1:
            best_effect = best['total_effect_6months']
            for candidate in all_combinations[1:]:
                # 效果不低于最优的98%但成本更低
                if candidate['total_effect_6months'] >= best_effect * 0.98 and \
                   candidate['total_cost_6months'] < best['total_cost_6months']:
                    best = candidate
        
        # 生成自然语言推荐
        recommendation = self._format_recommendation(best) if best else "无法找到可行方案"
        
        result = {
            'patient_type': patient_type,
            'patient_type_name': type_info['name'],
            'prefer_strategy': type_info['prefer_strategy'],
            'paper_conclusion': self._get_paper_conclusion(patient_type),
            'max_budget': max_budget,
            'optimal_plan': best,
            'recommendation': recommendation,
            'feasible_count': len(all_combinations),
            'constraints_summary': self._build_constraints_summary(patient_type, features or {})
        }
        
        return result
    
    def get_top_n_plans(
        self,
        patient_type: str,
        max_budget: int,
        features: Optional[Dict[str, float]] = None,
        n: int = 3
    ) -> List[Dict[str, Any]]:
        """获取前N个最优方案供选择"""
        all_combinations = self.generate_all_combinations(patient_type, max_budget, features)
        all_combinations.sort(key=lambda x: x['total_effect_6months'], reverse=True)
        
        # 去重并返回前n个
        result = []
        seen = set()
        for combo in all_combinations:
            key = (combo['treatment_key'], combo['intensity_key'], combo.get('frequency_name'))
            if key not in seen:
                seen.add(key)
                # 添加友好名称
                combo['treatment_name'] = combo['treatment']['name']
                if combo['exercise']:
                    combo['exercise_fullname'] = f"{combo['intensity_name']} {combo['frequency_name']}"
                else:
                    combo['exercise_fullname'] = "无运动方案"
                result.append(combo)
                if len(result) >= n:
                    break
        
        return result

    def _build_constraints_summary(self, patient_type: str, features: Dict[str, float]) -> Dict[str, Any]:
        """生成前端可展示的约束说明"""
        age = features.get('age')
        activity = features.get('activity_score')
        tan_score = features.get('tan_score_raw')
        allowed = self._allowed_intensities(features, self._get_constraints_for_type(patient_type))
        return {
            'source': 'MathorCup C题附表2、附表3',
            'tan_score_rule': self._tan_score_rule_text(tan_score),
            'age_rule': self._age_rule_text(age),
            'activity_rule': self._activity_rule_text(activity),
            'allowed_intensities': [self.exercise_plans[key]['name'] for key in allowed],
            'frequency_range': f"每周1-{self._get_constraints_for_type(patient_type).get('max_frequency_per_week', 10)}次",
            'cycle': '6个月，按24周计算',
            'budget_rule': '单人6个月总成本建议不超过2000元'
        }

    def _tan_score_rule_text(self, tan_score: Optional[float]) -> str:
        if tan_score is None:
            return '按痰湿积分匹配基础/中度/强化调理'
        if tan_score <= 58:
            return f'痰湿积分{tan_score:.0f}分，匹配基础调理'
        if tan_score <= 61:
            return f'痰湿积分{tan_score:.0f}分，匹配中度调理'
        return f'痰湿积分{tan_score:.0f}分，匹配强化调理'

    def _age_rule_text(self, age: Optional[float]) -> str:
        if age is None:
            return '按年龄限制最大运动强度'
        if age < 60:
            return f'{age:.0f}岁，可选择1/2/3级活动强度'
        if age < 80:
            return f'{age:.0f}岁，可选择1/2级活动强度'
        return f'{age:.0f}岁，仅建议1级活动强度'

    def _activity_rule_text(self, activity: Optional[float]) -> str:
        if activity is None:
            return '按活动量表总分限制最大运动强度'
        if activity < 40:
            return f'活动量表{activity:.0f}分，仅建议1级活动强度'
        if activity < 60:
            return f'活动量表{activity:.0f}分，可选择1/2级活动强度'
        return f'活动量表{activity:.0f}分，可选择1/2/3级活动强度'
    
    def _format_recommendation(self, plan: Dict[str, Any]) -> str:
        """格式化推荐建议为自然语言"""
        treatment = plan['treatment']
        lines = []
        
        lines.append(f"### 推荐方案\n")
        lines.append(f"**调理方案**: {treatment['name']}")
        lines.append(f"- 内容: {treatment['description']}")
        lines.append(f"- 月费用: **{treatment['cost_per_month']}** 元")
        
        if plan['exercise']:
            lines.append("")
            lines.append(f"**运动方案**: {plan['intensity_name']}, {plan['frequency_name']}, 每次{plan['duration_minutes']}分钟")
            lines.append(f"- 适用: {plan['exercise']['tolerance']}")
            lines.append(f"- 单次费用: **{plan['exercise']['cost_per_session']}** 元")
        
        lines.append("")
        lines.append(f"### 费用与效果")
        lines.append(f"- 6个月总成本: **{plan['total_cost_6months']:.0f}** 元")
        lines.append(f"- 预期总效果: **{plan['total_effect_6months']:.1f}** 单位 (痰湿积分改善预期)")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    print("=== 个体化干预方案优化 - 测试 ===")
    
    optimizer = InterventionOptimizer()
    
    # 测试三种类型
    for ptype in ['metabolic', 'obesity', 'function_limited']:
        print(f"\n-------- {ptype} --------")
        result = optimizer.optimize(ptype, 2000)
        print(f"类型: {result['patient_type_name']}")
        print(f"论文结论: {result['paper_conclusion']}")
        print(f"可行方案数: {result['feasible_count']}")
        if result['optimal_plan']:
            print(f"最优方案 - 6个月成本: {result['optimal_plan']['total_cost_6months']:.0f}, "
                  f"效果: {result['optimal_plan']['total_effect_6months']:.1f}")
