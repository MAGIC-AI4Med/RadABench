# Define the Tool class
import inspect
import random

def get_nested_attribute(data, key):
    """
    Recursively searches for a specific key within nested dictionaries.
    If the key is found, returns its value; otherwise, returns None.

    Parameters:
        data (dict): The nested dictionary to search.
        key (str): The key to look for.

    Returns:
        The value associated with the key, if found; otherwise, None.
    """
    # If data is a dictionary, check if the key is in this level
    if isinstance(data, dict):
        if key in data:
            return data[key]
        # Otherwise, iterate through each value in the dictionary
        for k, v in data.items():
            result = get_nested_attribute(v, key)
            if result is not None:
                return result
    # If data is a list, check each element (in case of nested lists)
    elif isinstance(data, list):
        for item in data:
            result = get_nested_attribute(item, key)
            if result is not None:
                return result
    # If key is not found, return None
    return None

class AnatomyError(Exception):
    pass
class ModalityError(Exception):
    pass
class SegmentationError(Exception):
    pass
class AnomalyError(Exception):
    pass
class DiagnosisError(Exception):
    pass
class InferenceError(Exception):
    pass
class BiomarkerError(Exception):
    pass
class IndicatorError(Exception):
    pass
class GenerationError(Exception):
    pass
class TreatmentError(Exception):
    pass

map_dict = {
    "Universal": ["Universal"],
    "Head and Neck": ['X-ray', 'CT', 'MRI', 'Ultrasound'],
    "Chest": ['X-ray', 'CT', 'MRI', 'Ultrasound'],
    "Breast": ['Mammography', 'MRI', 'Ultrasound'],
    "Abdomen and Pelvis": ['X-ray', 'CT', 'MRI', 'Ultrasound'],
    "Limb": ['X-ray', 'CT', 'MRI', 'Ultrasound'],
    "Spine": ['X-ray', 'CT', 'MRI'],
}

Error_dict = {
    "Anatomy Classifier": AnatomyError,
    "Modality Classifier": ModalityError,
    "Organ Segmentor": SegmentationError,
    "Anomaly Detector": AnomalyError,
    "Disease Diagnoser": DiagnosisError,
    "Disease Inferencer": InferenceError,
    "Biomarker Quantifier": BiomarkerError,
    "Indicator Evaluator": IndicatorError,
    "Report Generator": GenerationError,
    "Treatment Recommender": TreatmentError,
}

class Tool:
    def __init__(self, tool_count, category, **kwargs):
        self.name = f"TOOL{tool_count}"
        self.category = category
        # 初始化所有可能用到的属性
        self.property = None
        self.ability = None
        self.compulsory_input = []
        self.optional_input = []
        self.output = None
        self.lower_bound = None
        self.upper_bound = None
        self.step = None
        self.anatomy = None
        self.modality = None
        self.organs = None
        self.anomalies = None
        self.diseases = None
        self.biomarkers = None
        self.indicators = None
        self.kwargs = kwargs
        # 构建工具
        self.build_tool()
    
    def adjust_performance(self, base_lower, base_upper, supported_list=None):
        """
        根据supported列表长度调整性能值
        base_lower: 基础lower_bound值
        base_upper: 基础upper_bound值
        supported_list: 支持的元素列表
        """
        if not supported_list:
            return base_lower, base_upper
            
        # 目标最高性能
        target_upper = 0.95
        # 计算可提升空间
        performance_gap = target_upper - base_upper
        
        # 如果列表只有一个元素，直接返回最高性能
        if len(supported_list) == 1:
            return base_lower + performance_gap, target_upper
        
        # 根据元素个数计算提升比例
        # 元素越少，提升越多
        boost_ratio = max(0, (11 - len(supported_list)) / 10)
        performance_boost = performance_gap * boost_ratio
        
        return base_lower + performance_boost, base_upper + performance_boost

    def build_tool(self):
        if self.category == 'Anatomy Classifier':
            self.property = "Universal Anatomy Classifier"
            self.ability = "Determine the anatomy of the Image."
            self.compulsory_input = ["$Image$"]
            self.optional_input = []
            self.output = ["$Anatomy$"]
            self.lower_bound = 0.95
            self.upper_bound = 0.95
            self.step = 0.0

        elif self.category == 'Modality Classifier':
            self.property = "Universal Modality Classifier"
            self.ability = "Determine the modality of the Image."
            self.compulsory_input = ["$Image$"]
            self.optional_input = []
            self.output = ["$Modality$"]
            self.lower_bound = 0.95
            self.upper_bound = 0.95
            self.step = 0.0

        elif self.category == 'Organ Segmentor':
            if (self.kwargs['Anatomy'] == None and self.kwargs['Modality'] == None) or (self.kwargs['Anatomy'] == 'Universal' and self.kwargs['Modality'] == 'Universal'):
                self.property = "Universal Organ Segmentor"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.organs = self.kwargs['Supported']
                    self.ability = f"Given the modality and anatomy, segment the organs in the Image. Supported organs are {self.organs}."
                    base_lower, base_upper = 0.80, 0.80
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.organs)
                else:                
                    self.ability = "Given the modality and anatomy, segment all organs in the Image."
                    self.lower_bound = 0.80
                    self.upper_bound = 0.80
                self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]
                self.optional_input = []
                self.output = ["$OrganMask$", "$OrganObject$"]
                self.step = 0.0
                
            else:
                self.anatomy = self.kwargs['Anatomy']
                self.modality = self.kwargs['Modality']
                self.property = f"Organ Segmentor only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.organs = self.kwargs['Supported']
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, segment the organs. Supported organs are {self.organs}."
                    base_lower, base_upper = 0.85, 0.85
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.organs)
                else:
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, segment the organs."
                    self.lower_bound = 0.85
                    self.upper_bound = 0.85
                self.compulsory_input = ["$Image$"]
                self.optional_input = []
                self.output = ["$OrganMask$", "$OrganObject$"]
                self.step = 0.0

        elif self.category == 'Anomaly Detector':
            if (self.kwargs['Anatomy'] == None and self.kwargs['Modality'] == None) or (self.kwargs['Anatomy'] == 'Universal' and self.kwargs['Modality'] == 'Universal'):
                self.property = "Universal Anomaly Detector"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.anomalies = self.kwargs['Supported']
                    self.ability = f"Given the modality and anatomy, determine the location and type of abnormality. Supported anomalies are {self.anomalies}."
                    base_lower, base_upper = 0.75, 0.75
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.anomalies)
                else:
                    self.ability = "Given the modality and anatomy, determine the location and type of abnormality."
                    self.lower_bound = 0.75
                    self.upper_bound = 0.75
                self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]
                self.optional_input = []
                self.output = ["$AnomalyMask$", "$AnomalyObject$"]
                self.step = 0.0
            else:
                self.anatomy = self.kwargs['Anatomy']
                self.modality = self.kwargs['Modality']
                self.property = f"Anomaly Detector only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.anomalies = self.kwargs['Supported']
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, determine the location and type of abnormality. Supported anomalies are {self.anomalies}."
                    base_lower, base_upper = 0.80, 0.80
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.anomalies)
                else:
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, determine the location and type of abnormality."
                    self.lower_bound = 0.80
                    self.upper_bound = 0.80
                self.compulsory_input = ["$Image$"]
                self.optional_input = []
                self.output = ["$AnomalyMask$", "$AnomalyObject$"]
                self.step = 0.0

        elif self.category == 'Disease Diagnoser':
            if (self.kwargs['Anatomy'] == None and self.kwargs['Modality'] == None) or (self.kwargs['Anatomy'] == 'Universal' and self.kwargs['Modality'] == 'Universal'):
                self.property = "Universal Disease Diagnoser"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.diseases = self.kwargs['Supported']
                    self.ability = f"Given the modality and anatomy, diagnose the disease. Supported diseases are {self.diseases}."
                    base_lower, base_upper = 0.70, 0.70
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.diseases)
                else:
                    self.ability = "Given the modality and anatomy, diagnose the disease."
                    self.lower_bound = 0.70
                    self.upper_bound = 0.70
                self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]
                self.optional_input = []
                self.output = ["$Disease$"]
                self.step = 0.0
            else:
                self.anatomy = self.kwargs['Anatomy']
                self.modality = self.kwargs['Modality']
                self.property = f"Disease Diagnoser only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.diseases = self.kwargs['Supported']
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, diagnose the disease. Supported diseases are {self.diseases}."
                    base_lower, base_upper = 0.75, 0.75
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.diseases)
                else:
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, diagnose the disease."
                    self.lower_bound = 0.75
                    self.upper_bound = 0.75
                self.compulsory_input = ["$Image$"]
                self.optional_input = []
                self.output = ["$Disease$"]
                self.step = 0.0

        elif self.category == 'Disease Inferencer':
            self.property = "Universal Disease Inferencer"
            if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                self.diseases = self.kwargs['Supported']
                self.ability = f"Infer disease based on the patient's Information, organ segmentation and anomaly detection results. Supported diseases are {self.diseases}."
                base_lower, base_upper = 0.80, 0.80
                self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.diseases)
            else:
                self.ability = "Infer disease based on the patient's Information, organ segmentation and anomaly detection results."
                self.lower_bound = 0.80
                self.upper_bound = 0.80
            self.compulsory_input = ["$Image$", "$Information$", "$OrganMask$", "$OrganObject$", "$AnomalyMask$", "$AnomalyObject$"]
            self.optional_input = []
            self.output = ["$Disease$"]
            self.step = 0.0

        elif self.category == 'Biomarker Quantifier':
            if self.kwargs['type'] == 'Organ':
                self.property = "Universal Organ Biomarker Quantifier"
                self.type = "Organ"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.biomarkers = self.kwargs['Supported']
                    self.ability = f"Measure the organ biomarker of the Image. Supported biomarker dims are {self.biomarkers}"
                    base_lower, base_upper = 0.75, 0.80
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.biomarkers)
                else:
                    self.ability = "Measure the organ biomarker of the Image."
                    self.lower_bound = 0.75
                    self.upper_bound = 0.80
                self.compulsory_input = ["$Image$", "$OrganObject$", "$OrganMask$"]
                self.optional_input = ["$OrganDim$"]
                self.output = ["$OrganDim$", "$OrganQuant$"]
                self.step = 0.05

            elif self.kwargs['type'] == 'Anomaly':
                self.property = "Universal Anomaly Biomarker Quantifier"
                self.type = "Anomaly"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.biomarkers = self.kwargs['Supported']
                    self.ability = f"Measure the anomaly biomarker of the Image. Supported biomarker dims are {self.biomarkers}"
                    base_lower, base_upper = 0.75, 0.80
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.biomarkers)
                else:
                    self.ability = "Measure the anomaly biomarker of the Image."
                    self.lower_bound = 0.75
                    self.upper_bound = 0.80
                self.compulsory_input = ["$Image$", "$AnomalyObject$", "$AnomalyMask$"]
                self.optional_input = ["$AnomalyDim$"]
                self.output = ["$AnomalyDim$", "$AnomalyQuant$"]
                self.step = 0.05

        elif self.category == 'Indicator Evaluator':
            if self.kwargs['type'] == 'Organ':
                self.property = "Universal Organ Indicator Evaluator"
                self.type = "Organ"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.indicators = self.kwargs['Supported']
                    self.ability = f"Evaluate the indicator based on the patient's Information and organ segmentation. Supported indicators are {self.indicators}"
                    base_lower, base_upper = 0.80, 0.80
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.indicators)
                else:
                    self.ability = "Evaluate the indicator based on the patient's Information and organ segmentation"
                    self.lower_bound = 0.80
                    self.upper_bound = 0.80
                self.compulsory_input = ["$Image$", "$Information$", "$IndicatorName$", "$OrganObject$", "$OrganDim$", "$OrganQuant$"]
                self.optional_input = []
                self.output = ["$IndicatorValue$"]
                self.step = 0.0
            elif self.kwargs['type'] == 'Anomaly':
                self.property = "Universal Anomaly Indicator Evaluator"
                self.type = "Anomaly"
                if 'Supported' in self.kwargs and self.kwargs['Supported'] != [] and self.kwargs['Supported'] != None:
                    self.indicators = self.kwargs['Supported']
                    self.ability = f"Evaluate the indicator based on the patient's Information and anomaly detection. Supported indicators are {self.indicators}"
                    base_lower, base_upper = 0.80, 0.80
                    self.lower_bound, self.upper_bound = self.adjust_performance(base_lower, base_upper, self.indicators)
                else:
                    self.ability = "Evaluate the indicator based on the patient's Information and anomaly detection"
                    self.lower_bound = 0.80
                    self.upper_bound = 0.80
                self.compulsory_input = ["$Image$", "$Information$", "$IndicatorName$", "$AnomalyObject$", "$AnomalyDim$", "$AnomalyQuant$"]
                self.optional_input = []
                self.output = ["$IndicatorValue$"]
                self.step = 0.0

        elif self.category == 'Report Generator':
            if (self.kwargs['Anatomy'] == None and self.kwargs['Modality'] == None) or (self.kwargs['Anatomy'] == 'Universal' and self.kwargs['Modality'] == 'Universal'):
                if self.kwargs['type'] == 'Basic':
                    self.property = "Universal Report Generator with Image Only"
                    self.ability = "Given the modality and anatomy, generate a basic radiology report from the input image."
                    self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]   
                    self.optional_input = []
                    self.output = ["$Report$"]
                    self.lower_bound = 0.35
                    self.upper_bound = 0.35
                    self.step = 0.0
                elif self.kwargs['type'] == 'Text':
                    self.property = "Universal Report Generator with Text"
                    self.ability = "Given the modality, anatomy and any other text information, generate a radiology report from the input image."
                    self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]   
                    self.optional_input = ["$Information$", "$OrganObject$", "$AnomalyObject$", "$Disease$", "$OrganDim$", "$OrganQuant$", "$AnomalyDim$", "$AnomalyQuant$", "$IndicatorName$", "$ValueName$"]
                    self.output = ["$Report$"]
                    self.lower_bound = 0.35
                    self.upper_bound = 0.65
                    self.step = 0.03
                elif self.kwargs['type'] == 'Mask':
                    self.property = "Universal Report Generator with Mask"
                    self.ability = "Given the modality, anatomy and any other organ/anomaly masks and labels, generate a radiology report from the input image.",
                    self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]   
                    self.optional_input = ["$OrganMask$", "$OrganObject$", "$AnomalyMask$", "$AnomalyObject$"]
                    self.output = ["$Report$"]
                    self.lower_bound = 0.35
                    self.upper_bound = 0.71
                    self.step = 0.09
                elif self.kwargs['type'] == 'Text and Mask':
                    self.property = "Universal Report Generator with Text and Mask"
                    self.ability = "Given the modality, anatomy, any other text information and organ/anomaly masks and labels, generate a radiology report from the input image."
                    self.compulsory_input = ["$Image$", "$Anatomy$", "$Modality$"]   
                    self.optional_input = ["$Information$", "$OrganObject$", "$AnomalyObject$", "$Disease$", "$OrganDim$", "$OrganQuant$", "$AnomalyDim$", "$AnomalyQuant$", "$IndicatorName$", "$ValueName$", "$OrganMask$", "$AnomalyMask$"]
                    self.output = ["$Report$"]
                    self.lower_bound = 0.35
                    self.upper_bound = 0.83
                    self.step = 0.04

            else:
                self.anatomy = self.kwargs['Anatomy']
                self.modality = self.kwargs['Modality']
                if self.kwargs['type'] == 'Basic':
                    self.property = f"Report Generator only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image with Image Only"
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, generate a basic radiology report."
                    self.compulsory_input = ["$Image$"]
                    self.optional_input = []
                    self.output = ["$Report$"]
                    self.lower_bound = 0.40
                    self.upper_bound = 0.40
                    self.step = 0.0
                elif self.kwargs['type'] == 'Text':
                    self.property = f"Report Generator only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image with Text"
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image and any other text information, generate a radiology report."
                    self.compulsory_input = ["$Image$"]
                    self.optional_input = ["$Information$", "$OrganObject$", "$AnomalyObject$", "$Disease$", "$OrganDim$", "$OrganQuant$", "$AnomalyDim$", "$AnomalyQuant$", "$IndicatorName$", "$ValueName$"]
                    self.output = ["$Report$"]
                    self.lower_bound = 0.40
                    self.upper_bound = 0.70
                    self.step = 0.03
                elif self.kwargs['type'] == 'Mask':
                    self.property = f"Report Generator only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image with Mask"
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image and any other organ/anomaly masks and labels, generate a radiology report."
                    self.compulsory_input = ["$Image$"]
                    self.optional_input = ["$OrganMask$", "$OrganObject$", "$AnomalyMask$", "$AnomalyObject$"]
                    self.output = ["$Report$"]
                    self.lower_bound = 0.40
                    self.upper_bound = 0.76
                    self.step = 0.09
                elif self.kwargs['type'] == 'Text and Mask':
                    self.property = f"Report Generator only suitable for {self.kwargs['Anatomy']} {self.kwargs['Modality']} image with Text and Mask"
                    self.ability = f"Given the {self.kwargs['Anatomy']} {self.kwargs['Modality']} Image, any other text information and organ/anomaly masks and labels, generate a radiology report."
                    self.compulsory_input = ["$Image$"]
                    self.optional_input = ["$Information$", "$OrganObject$", "$AnomalyObject$", "$Disease$", "$OrganDim$", "$OrganQuant$", "$AnomalyDim$", "$AnomalyQuant$", "$IndicatorName$", "$ValueName$", "$OrganMask$", "$AnomalyMask$"]
                    self.output = ["$Report$"]
                    self.lower_bound = 0.40
                    self.upper_bound = 0.88
                    self.step = 0.04

        elif self.category == 'Treatment Recommender':
            self.property = "Universal Treatment Recommender"
            self.ability = "Recommend treatment based on the patient's Information and Report"
            self.compulsory_input = ["$Information$", "$Report$"]
            self.optional_input = ["$Disease$", "$OrganDim$", "$OrganQuant$", "$AnomalyDim$", "$AnomalyQuant$", "$IndicatorName$", "$ValueName$"]
            self.output = ["$Treatment$"]
            self.lower_bound = 0.80
            self.upper_bound = 0.98
            self.step = 0.02
        else:
            raise ValueError(f"Category {self.category} is not defined.")
        
    def to_info_dict(self):
        return {
            "Name": self.name,
            "Category": self.category,
            "Property": self.property,
            "Ability": self.ability,
            "Compulsory Input": self.compulsory_input,
            "Optional Input": self.optional_input,
            "Output": self.output,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "step": self.step,
            "Performance": f"Score from {self.lower_bound} to {self.upper_bound}, increases with optional inputs",
            "Anatomy": self.anatomy,
            "Modality": self.modality,
            "Organs": self.organs,
            "Anomalies": self.anomalies,
            "Diseases": self.diseases,
            "Biomarkers": self.biomarkers,
            "Indicators": self.indicators,
            "type": self.type if hasattr(self, 'type') else None
        }

    def to_des_dict(self):
        return {
            "Name": self.name,
            "Category": self.category,
            "Ability": self.ability,
            "Property": self.property,
            "Compulsory Input": self.compulsory_input,
            "Optional Input": self.optional_input,
            "Output": self.output,
            "Performance": f"Score from {self.lower_bound} to {self.upper_bound}, increases with optional inputs"
        }
    
    def to_api_func(self):
        def tool_func(input, value_dict, score_dict, fixed_dict):
            # 检查解剖部位和成像模态的匹配
            if self.anatomy and self.modality and self.anatomy != 'Universal' and self.modality != 'Universal':
                if value_dict.get('$Anatomy$') != self.anatomy or value_dict.get('$Modality$') != self.modality:
                    raise Error_dict[self.category](
                        f"{self.property}: Requires {self.anatomy} {self.modality}, but got "
                        f"{value_dict.get('$Anatomy$')} {value_dict.get('$Modality$')}"
                    )

            # 验证必需输入
            for item in self.compulsory_input:
                if item not in input:
                    raise Error_dict[self.category](
                        f"{self.property}: Missing compulsory input: {item}"
                    )
                if item not in value_dict or item not in score_dict:
                    raise Error_dict[self.category](
                        f"{self.property}: Missing value or score for compulsory input: {item}"
                    )

            # 计算基础性能分数（基于必需输入）
            current_coef = self.lower_bound
            min_compulsory_score = min(score_dict[item] for item in self.compulsory_input)
            current_coef *= min_compulsory_score

            # 处理可选输入带来的性能提升
            optional_inputs = set(input) - set(self.compulsory_input)
            for item in optional_inputs:
                if item not in self.optional_input:
                    raise Error_dict[self.category](
                        f"{self.property}: Invalid optional input: {item}"
                    )
                if item in value_dict and item in score_dict:
                    performance_boost = self.step * score_dict[item]
                    current_coef = min(current_coef + performance_boost, self.upper_bound)

            # 设置输出值和分数
            for item in self.output:
                value_dict[item] = f"PLACEHOLDER_{item}"
                score_dict[item] = current_coef

            # 验证固定分数约束
            for item, fixed_score in fixed_dict.items():
                if score_dict[item] > fixed_score:
                    raise Error_dict[self.category](
                        f"{self.property}: Score {score_dict[item]} exceeds fixed score {fixed_score} for {item}"
                    )
                score_dict[item] = fixed_score

            return value_dict, score_dict

        return tool_func

class ToolManager_IO:
    def __init__(self, combined_cases=None, case=None, type=None):
        self.toolbox = {}
        self.tooldes = {}
        self.toolinfo = {}
        self.toolcount = 0
        if case != None:
            self.case = case
            self.anatomy = case['Anatomy']
            self.modality = case['Modality']

    def build_tools(self):
        # build anatomy tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build modality tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build organ segmentor tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
        # build anomaly detector tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
        # build disease diagnoser tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
        #build disease inferencer tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build biomarker quantifier tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build indicator evaluator tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build report generator tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Basic')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Text')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Mask')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Text and Mask')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
        # build treatment recommender tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
    # 现有的build_tools方法保持不变
        
    def build_tools_toy(self):
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build modality tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build organ segmentor tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', Anatomy='Universal', Modality='Universal')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        #build anomaly detector tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', Anatomy='Universal', Modality='Universal')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build disease diagnoser tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', Anatomy='Universal', Modality='Universal')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        #build disease inferencer tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build biomarker quantifier tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build indicator evaluator tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build report generator tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy='Universal', Modality='Universal', type='Text and Mask')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build treatment recommender tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

    
    def build_tools_basic(self):
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build modality tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build organ segmentor tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                break
            break
        # build anomaly detector tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                break
            break
        # build disease diagnoser tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                break
            break
        #build disease inferencer tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build biomarker quantifier tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build indicator evaluator tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        # build report generator tool
        for anatomy in map_dict:
            for modality in map_dict[anatomy]:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Basic')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Text')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Mask')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', Anatomy=anatomy, Modality=modality, type='Text and Mask')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                break
            break
        # build treatment recommender tool
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

    def build_tools_scale(self):
        # 构建解剖分类器和模态分类器（各一个）
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # 为organ segmentor, anomaly detector和disease diagnoser选择工具
        for category in ['Organ Segmentor', 'Anomaly Detector', 'Disease Diagnoser']:
            # 同模态同部位的工具
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category=category, 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

            # 同模态不同部位的工具
            same_modality_anatomies = [anat for anat in map_dict.keys() 
                                    if self.modality in map_dict[anat] and anat != self.anatomy]
            selected_anatomies = random.sample(same_modality_anatomies, 
                                            min(2, len(same_modality_anatomies)))
            for anatomy in selected_anatomies:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category=category, 
                        Anatomy=anatomy, Modality=self.modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

            # 同部位不同模态的工具
            same_anatomy_modalities = [mod for mod in map_dict[self.anatomy] 
                                    if mod != self.modality]
            selected_modalities = random.sample(same_anatomy_modalities, 
                                            min(2, len(same_anatomy_modalities)))
            for modality in selected_modalities:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category=category, 
                        Anatomy=self.anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

        # Disease Inferencer（一个）
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Biomarker Quantifier（两个）
        for type_ in ['Organ', 'Anomaly']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Indicator Evaluator（两个）
        for type_ in ['Organ', 'Anomaly']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Report Generator
        # 同模态同部位的4个工具
        for type_ in ['Basic', 'Text', 'Mask', 'Text and Mask']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Report Generator', 
                    Anatomy=self.anatomy, Modality=self.modality, type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # 同模态不同部位的4个工具
        same_modality_anatomies = [anat for anat in map_dict.keys() 
                                if self.modality in map_dict[anat] and anat != self.anatomy]
        if same_modality_anatomies:
            selected_anatomies = random.sample(same_modality_anatomies, 
                                            min(4, len(same_modality_anatomies)))
            for anatomy in selected_anatomies:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', 
                        Anatomy=anatomy, Modality=self.modality, 
                        type=random.choice(['Basic', 'Text', 'Mask', 'Text and Mask']))
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

        # 同部位不同模态的4个工具
        same_anatomy_modalities = [mod for mod in map_dict[self.anatomy] 
                                if mod != self.modality]
        if same_anatomy_modalities:
            selected_modalities = random.sample(same_anatomy_modalities, 
                                            min(4, len(same_anatomy_modalities)))
            for modality in selected_modalities:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator', 
                        Anatomy=self.anatomy, Modality=modality, 
                        type=random.choice(['Basic', 'Text', 'Mask', 'Text and Mask']))
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

        # Treatment Recommender（一个）
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

    def build_tools_denyl1(self, category):
        if category not in ['Anatomy Classifier', 'Modality Classifier', 'Organ Segmentor', 'Anomaly Detector', 'Disease Judge', 'Disease Inferencer', 
                            'Organ Biomarker Quantifier', 'Anomaly Biomarker Quantifier', 'Organ Indicator Evaluator', 
                            'Anomaly Indicator Evaluator', 'Report Generator', 'Treatment Recommender']:
            raise ValueError(f"Category {category} is not defined.")
        # 初始化anatomy和modality分类器
        if category != 'Anatomy Classifier':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        
        if category != 'Modality Classifier':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # 对于Organ Segmentor, Anomaly Detector, Disease Diagnoser
        if category != 'Organ Segmentor':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if category != 'Anomaly Detector':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if category != 'Disease Judge':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        
        # if category != 'Disease Inferencer':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # 对于Biomarker Quantifiers和Indicator Evaluators
        if category != 'Organ Biomarker Quantifier':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if category != 'Anomaly Biomarker Quantifier':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if category != 'Organ Indicator Evaluator':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if category != 'Anomaly Indicator Evaluator':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # 对于Report Generator
        if category != 'Report Generator':
            # 随机选择两个universal类型
            types = ['Basic', 'Text', 'Mask', 'Text and Mask']
            selected_types = random.sample(types, 2)
            
            for type_ in selected_types:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator',
                        Anatomy='Universal', Modality='Universal', type=type_)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

            # 随机选择两个specific类型
            selected_types = random.sample(types, 2)
            for type_ in selected_types:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category='Report Generator',
                        Anatomy=self.anatomy, Modality=self.modality, type=type_)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

        # 对于Treatment Recommender
        if category != 'Treatment Recommender':
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

    def build_tools_denyl2(self, missing_category):
        if missing_category not in ['Organ', 'Anomaly', 'Disease', 'Report']:
            raise ValueError(f"Category {missing_category} is not defined.")

        # Step 1: Anatomy Classifier
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Step 2: Modality Classifier
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Helper function to get valid combinations
        def get_valid_combinations():
            valid_combinations = []
            for anatomy in map_dict:
                if anatomy != self.anatomy:
                    for modality in map_dict[anatomy]:
                        if (anatomy != 'Universal' and modality != 'Universal' and 
                            (anatomy != self.anatomy or modality != self.modality)):
                            valid_combinations.append((anatomy, modality))
            return valid_combinations

        # Step 3-5: Handle Organ Segmentor, Anomaly Detector, Disease Diagnoser
        categories_map = {
            'Organ': 'Organ Segmentor',
            'Anomaly': 'Anomaly Detector',
            'Disease': 'Disease Diagnoser'
        }

        for category in ['Organ', 'Anomaly', 'Disease']:
            if category == missing_category:
                # Get two random valid combinations
                valid_combinations = get_valid_combinations()
                if len(valid_combinations) >= 2:
                    selected_combinations = random.sample(valid_combinations, 2)
                    for anatomy, modality in selected_combinations:
                        self.toolcount += 1
                        tool = Tool(tool_count=self.toolcount, 
                                category=categories_map[category],
                                Anatomy=anatomy, 
                                Modality=modality)
                        self.toolinfo[tool.name] = tool.to_info_dict()
                        self.tooldes[tool.name] = tool.to_des_dict()
                        self.toolbox[tool.name] = tool.to_api_func()
            else:
                # Create Universal tool
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, 
                        category=categories_map[category],
                        Anatomy='Universal', 
                        Modality='Universal')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

                # Create case-specific tool
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, 
                        category=categories_map[category],
                        Anatomy=self.anatomy, 
                        Modality=self.modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

        # Step 6: Add Biomarker Quantifiers
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Step 7: Add Indicator Evaluators
        for type_ in ['Organ', 'Anomaly']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 8: Handle Report Generator
        if missing_category == 'Report':
            valid_combinations = get_valid_combinations()
            if len(valid_combinations) >= 2:
                selected_combinations = random.sample(valid_combinations, 2)
                for anatomy, modality in selected_combinations:
                    for type_ in ['Basic', 'Text and Mask']:
                        self.toolcount += 1
                        tool = Tool(tool_count=self.toolcount, 
                                category='Report Generator',
                                Anatomy=anatomy, 
                                Modality=modality, 
                                type=type_)
                        self.toolinfo[tool.name] = tool.to_info_dict()
                        self.tooldes[tool.name] = tool.to_des_dict()
                        self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Create Universal tools
            for type_ in ['Basic', 'Text and Mask']:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, 
                        category='Report Generator',
                        Anatomy='Universal', 
                        Modality='Universal', 
                        type=type_)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

            # Create case-specific tools
            for type_ in ['Basic', 'Text and Mask']:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, 
                        category='Report Generator',
                        Anatomy=self.anatomy, 
                        Modality=self.modality, 
                        type=type_)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()

        # Step 9: Add Treatment Recommender
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

    def build_tools_denyl2_complex(self, missing_category):
        if missing_category not in ['Organ', 'Anomaly', 'Disease', 'Report']:
            raise ValueError(f"Category {missing_category} is not defined.")
        # Keep these tools unchanged
        # Anatomy Classifier
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        # Modality Classifier
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        # Biomarker Quantifier (both types)
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        # Indicator Evaluator (both types)
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        # Treatment Recommender
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Handle the four special categories
        categories = {
            'Organ': 'Organ Segmentor',
            'Anomaly': 'Anomaly Detector', 
            'Disease': 'Disease Diagnoser',
            'Report': 'Report Generator'
        }
        
        # For the missing category
        if missing_category != 'Report':
            # Get all possible tools except universal and case-matching ones
            available_tools = []
            for anatomy in map_dict:
                for modality in map_dict[anatomy]:
                    if (anatomy != 'Universal' and modality != 'Universal' and 
                        (anatomy != self.anatomy or modality != self.modality)):
                        available_tools.append((anatomy, modality))
            
            # Randomly select 2 tools
            selected_tools = random.sample(available_tools, 2)
            for anatomy, modality in selected_tools:
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category=categories[missing_category], 
                        Anatomy=anatomy, Modality=modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
        
        else:  # Handle Report Generator separately
            # Get available non-universal tools
            available_tools = []
            for anatomy in map_dict:
                for modality in map_dict[anatomy]:
                    if (anatomy != 'Universal' and modality != 'Universal' and 
                        (anatomy != self.anatomy or modality != self.modality)):
                        available_tools.append((anatomy, modality))
            
            # Randomly select 2 tools
            selected_tools = random.sample(available_tools, 2)
            types = ['Basic', 'Text', 'Mask', 'Text and Mask']
            for anatomy, modality in selected_tools:
                # Randomly select 2 types for each tool
                selected_types = random.sample(types, 2)
                for type_ in selected_types:
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category='Report Generator',
                            Anatomy=anatomy, Modality=modality, type=type_)
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()
        
        # For the other three categories
        other_categories = [cat for cat in ['Organ', 'Anomaly', 'Disease', 'Report'] 
                        if cat != missing_category]
        
        for category in other_categories:
            if category != 'Report':
                # Add universal tool
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category=categories[category],
                        Anatomy='Universal', Modality='Universal')
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
                
                # Add case-matching tool
                self.toolcount += 1
                tool = Tool(tool_count=self.toolcount, category=categories[category],
                        Anatomy=self.anatomy, Modality=self.modality)
                self.toolinfo[tool.name] = tool.to_info_dict()
                self.tooldes[tool.name] = tool.to_des_dict()
                self.toolbox[tool.name] = tool.to_api_func()
            
            else:  # Handle Report Generator
                # Add 2 random universal tools
                types = ['Basic', 'Text', 'Mask', 'Text and Mask']
                selected_types = random.sample(types, 2)
                for type_ in selected_types:
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category='Report Generator',
                            Anatomy='Universal', Modality='Universal', type=type_)
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()
                
                # Add 2 random case-matching tools
                selected_types = random.sample(types, 2)
                for type_ in selected_types:
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category='Report Generator',
                            Anatomy=self.anatomy, Modality=self.modality, type=type_)
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()

    def build_tools_denyl3(self, type, entity_dict):
        if type not in ['Organ', 'Anomaly', 'Disease', 'OrganBiomarker', 'AnomalyBiomarker', 'Indicator']:
            raise ValueError(f"Type {type} is not defined.")

        # Step 1: Anatomy Classifier
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Step 2: Modality Classifier
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Step 3: Organ Segmentor
        if type == 'Organ':
            available_organs = [org for org in entity_dict['organ_list'] if org not in [get_nested_attribute(self.case, 'OrganObject')]]
            supported_organs = random.sample(available_organs, min(random.randint(3, 5), len(available_organs)))
            
            # Add tools with supported organs
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy='Universal', Modality='Universal', Supported=supported_organs)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_organs)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Add normal tools
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 4: Anomaly Detector
        if type == 'Anomaly':
            available_anomalies = [anom for anom in entity_dict['anomaly_list'] if anom not in [get_nested_attribute(self.case, 'AnomalyObject')]]
            supported_anomalies = random.sample(available_anomalies, min(random.randint(3, 5), len(available_anomalies)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy='Universal', Modality='Universal', Supported=supported_anomalies)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_anomalies)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 5: Disease Diagnoser and Inferencer
        if type == 'Disease':
            available_diseases = [dis for dis in entity_dict['disease_list'] if dis not in [get_nested_attribute(self.case, 'Disease')]]
            supported_diseases = random.sample(available_diseases, min(random.randint(3, 5), len(available_diseases)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy='Universal', Modality='Universal', Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Inferencer', Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()

        else:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 7: Biomarker Quantifier
        if type == 'OrganBiomarker':
            available_dims = [dim for dim in entity_dict['organ_dim_list'] if dim not in [get_nested_attribute(self.case, 'OrganDim')]]
            supported_dims = random.sample(available_dims, min(random.randint(3, 5), len(available_dims)))
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', 
                    type='Organ', Supported=supported_dims)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if type == 'AnomalyBiomarker':
            available_dims = [dim for dim in entity_dict['anomaly_dim_list'] if dim not in [get_nested_attribute(self.case, 'AnomalyDim')]]
            supported_dims = random.sample(available_dims, min(random.randint(3, 5), len(available_dims)))
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', 
                    type='Anomaly', Supported=supported_dims)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 8: Indicator Evaluator
        if type == 'Indicator':
            available_indicators = [ind for ind in entity_dict['indicator_list'] if ind not in [get_nested_attribute(self.case, 'Name')]]
            supported_indicators = random.sample(available_indicators, min(random.randint(3, 5), len(available_indicators)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', 
                    type='Organ', Supported=supported_indicators)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', 
                    type='Anomaly', Supported=supported_indicators)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 9: Report Generator (只保留Basic和Text and Mask两种类型)
        # Universal版本
        for type_ in ['Basic', 'Text and Mask']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Report Generator',
                    Anatomy='Universal', Modality='Universal', type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Case-specific版本
        for type_ in ['Basic', 'Text and Mask']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Report Generator',
                    Anatomy=self.anatomy, Modality=self.modality, type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Step 10: Treatment Recommender
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

    def build_tools_denyl3_complex(self, type, entity_dict):
        if type not in ['Organ', 'Anomaly', 'Disease', 'OrganBiomarker', 'AnomalyBiomarker', 'Indicator']:
            raise ValueError(f"Type {type} is not defined.")
        # 初始化anatomy和modality工具
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # 为type对应的工具设置障碍
        if type == 'Organ':
            # 从organ_list中排除当前case的organ并随机选择3-5个
            # available_organs = [org for org in entity_dict['organ_list'] if org not in self.case.get('Organs', [])]
            available_organs = [org for org in entity_dict['organ_list'] if org not in [get_nested_attribute(self.case, 'OrganObject')]]
            # print(available_organs)
            supported_organs = random.sample(available_organs, min(random.randint(3, 5), len(available_organs)))
            
            # 添加universal工具
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy='Universal', Modality='Universal', Supported=supported_organs)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            # 添加当前case匹配的工具（如果适用）
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_organs)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        elif type == 'Anomaly':
            # available_anomalies = [anom for anom in entity_dict['anomaly_list'] if anom not in self.case.get('Anomalies', [])]
            available_anomalies = [anom for anom in entity_dict['anomaly_list'] if anom not in [get_nested_attribute(self.case, 'AnomalyObject')]]
            # print(available_anomalies)
            supported_anomalies = random.sample(available_anomalies, min(random.randint(3, 5), len(available_anomalies)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy='Universal', Modality='Universal', Supported=supported_anomalies)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_anomalies)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        elif type == 'Disease':
            # available_diseases = [dis for dis in entity_dict['disease_list'] if dis not in self.case.get('Diseases', [])]
            available_diseases = [dis for dis in entity_dict['disease_list'] if dis not in [get_nested_attribute(self.case, 'Disease')]]
            # print(available_diseases)
            supported_diseases = random.sample(available_diseases, min(random.randint(3, 5), len(available_diseases)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy='Universal', Modality='Universal', Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        elif type == 'OrganBiomarker':
            # available_dims = [dim for dim in entity_dict['organ_dim_list'] if dim not in self.case.get('OrganDims', [])]
            available_dims = [dim for dim in entity_dict['organ_dim_list'] if dim not in [get_nested_attribute(self.case, 'OrganDim')]]
            supported_dims = random.sample(available_dims, min(random.randint(3, 5), len(available_dims)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', 
                    type='Organ', Supported=supported_dims)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        elif type == 'AnomalyBiomarker':
            # available_dims = [dim for dim in entity_dict['anomaly_dim_list'] if dim not in self.case.get('AnomalyDims', [])]
            available_dims = [dim for dim in entity_dict['anomaly_dim_list'] if dim not in [get_nested_attribute(self.case, 'AnomalyDim')]]
            # print(available_dims)
            supported_dims = random.sample(available_dims, min(random.randint(3, 5), len(available_dims)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', 
                    type='Anomaly', Supported=supported_dims)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        elif type == 'Indicator':
            # available_indicators = [ind for ind in entity_dict['indicator_list'] if ind not in self.case.get('Indicators', [])]
            available_indicators = [ind for ind in entity_dict['indicator_list'] if ind not in [get_nested_attribute(self.case, 'Name')]]
            # print(available_indicators)
            supported_indicators = random.sample(available_indicators, min(random.randint(3, 5), len(available_indicators)))
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', 
                    type='Organ', Supported=supported_indicators)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', 
                    type='Anomaly', Supported=supported_indicators)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # 为其他工具添加universal和case-specific的版本
        categories_map = {
            'Organ': 'Organ Segmentor',
            'Anomaly': 'Anomaly Detector',
            'Disease': 'Disease Diagnoser',
            'OrganBiomarker': 'Biomarker Quantifier',
            'AnomalyBiomarker': 'Biomarker Quantifier',
            'Indicator': 'Indicator Evaluator'
        }
        
        for category in categories_map:
            if category != type:
                if category in ['Organ', 'Anomaly', 'Disease']:
                    # 添加universal版本
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category=categories_map[category],
                            Anatomy='Universal', Modality='Universal')
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()
                    
                    # 添加case-specific版本
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category=categories_map[category],
                            Anatomy=self.anatomy, Modality=self.modality)
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()
                elif category == 'OrganBiomarker':
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()
                elif category == 'AnomalyBiomarker':
                    self.toolcount += 1
                    tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
                    self.toolinfo[tool.name] = tool.to_info_dict()
                    self.tooldes[tool.name] = tool.to_des_dict()
                    self.toolbox[tool.name] = tool.to_api_func()
                elif category == 'Indicator':
                    for indicator_type in ['Organ', 'Anomaly']:
                        self.toolcount += 1
                        tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type=indicator_type)
                        self.toolinfo[tool.name] = tool.to_info_dict()
                        self.tooldes[tool.name] = tool.to_des_dict()
                        self.toolbox[tool.name] = tool.to_api_func()

        # 添加Report Generator工具
        # Universal版本
        for type_ in ['Basic', 'Text', 'Mask', 'Text and Mask']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Report Generator',
                    Anatomy='Universal', Modality='Universal', type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Case-specific版本
        for type_ in ['Basic', 'Text', 'Mask', 'Text and Mask']:
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Report Generator',
                    Anatomy=self.anatomy, Modality=self.modality, type=type_)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # 添加Treatment Recommender工具
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
    
    def build_tools_performance(self, type, entity_dict):
        if type not in ['Organ', 'Anomaly', 'Disease', 'OrganBiomarker', 'AnomalyBiomarker', 'Indicator']:
            raise ValueError(f"Type {type} is not defined.")
        # 初始化anatomy和modality工具
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Anatomy Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()
        
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Modality Classifier')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # 根据type设置对应工具的3个版本，其他工具设置2个版本
        if type == 'Organ':
            # Organ Segmentor - 3个版本
            # 1. Universal
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            # 2. 同anatomy同modality无限制
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            # 3. 同anatomy同modality有限制
            current_organs = [get_nested_attribute(self.case, 'OrganObject')]
            # print(current_organs)
            other_organs = [org for org in entity_dict['organ_list'] if org not in current_organs]
            additional_organs = random.sample(other_organs, min(random.randint(3, 4), len(other_organs)))
            supported_organs = current_organs + additional_organs
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_organs)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Organ Segmentor - 2个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Organ Segmentor', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if type == 'Anomaly':
            # Anomaly Detector - 3个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            current_anomalies = [get_nested_attribute(self.case, 'AnomalyObject')]
            other_anomalies = [anom for anom in entity_dict['anomaly_list'] if anom not in current_anomalies]
            additional_anomalies = random.sample(other_anomalies, min(random.randint(3, 4), len(other_anomalies)))
            supported_anomalies = current_anomalies + additional_anomalies
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_anomalies)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Anomaly Detector - 2个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Anomaly Detector', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if type == 'Disease':
            # Disease Diagnoser - 3个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            current_diseases = [get_nested_attribute(self.case, 'Disease')]
            other_diseases = [dis for dis in entity_dict['disease_list'] if dis not in current_diseases]
            additional_diseases = random.sample(other_diseases, min(random.randint(3, 4), len(other_diseases)))
            supported_diseases = current_diseases + additional_diseases
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality, Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Inferencer', Supported=supported_diseases)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
        else:
            # Disease Diagnoser - 2个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy='Universal', Modality='Universal')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Diagnoser', 
                    Anatomy=self.anatomy, Modality=self.modality)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Disease Inferencer')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if type == 'OrganBiomarker':
            # Organ Biomarker Quantifier - 2个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            current_dims = [get_nested_attribute(self.case, 'OrganDim')]
            other_dims = [dim for dim in entity_dict['organ_dim_list'] if dim not in current_dims]
            additional_dims = random.sample(other_dims, min(random.randint(3, 4), len(other_dims)))
            supported_dims = current_dims + additional_dims
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', 
                    type='Organ', Supported=supported_dims)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Organ Biomarker Quantifier - 1个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if type == 'AnomalyBiomarker':
            # Anomaly Biomarker Quantifier - 2个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            current_dims = [get_nested_attribute(self.case, 'AnomalyDim')]
            other_dims = [dim for dim in entity_dict['anomaly_dim_list'] if dim not in current_dims]
            additional_dims = random.sample(other_dims, min(random.randint(3, 4), len(other_dims)))
            supported_dims = current_dims + additional_dims
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', 
                    type='Anomaly', Supported=supported_dims)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Anomaly Biomarker Quantifier - 1个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Biomarker Quantifier', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        if type == 'Indicator':
            # Indicator Evaluator - 各2个版本
            # Organ Indicator
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            current_indicators = [get_nested_attribute(self.case, 'Name')]
            other_indicators = [ind for ind in entity_dict['indicator_list'] if ind not in current_indicators]
            additional_indicators = random.sample(other_indicators, min(random.randint(3, 4), len(other_indicators)))
            supported_indicators = current_indicators + additional_indicators
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', 
                    type='Organ', Supported=supported_indicators)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            # Anomaly Indicator
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', 
                    type='Anomaly', Supported=supported_indicators)
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
        else:
            # Indicator Evaluator - 各1个版本
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Organ')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()
            
            self.toolcount += 1
            tool = Tool(tool_count=self.toolcount, category='Indicator Evaluator', type='Anomaly')
            self.toolinfo[tool.name] = tool.to_info_dict()
            self.tooldes[tool.name] = tool.to_des_dict()
            self.toolbox[tool.name] = tool.to_api_func()

        # Report Generator - 只保留Text and Mask类型
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Report Generator',
                Anatomy='Universal', Modality='Universal', type='Text and Mask')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Report Generator',
                Anatomy=self.anatomy, Modality=self.modality, type='Text and Mask')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

        # Treatment Recommender - 1个版本
        self.toolcount += 1
        tool = Tool(tool_count=self.toolcount, category='Treatment Recommender')
        self.toolinfo[tool.name] = tool.to_info_dict()
        self.tooldes[tool.name] = tool.to_des_dict()
        self.toolbox[tool.name] = tool.to_api_func()

    def calculate_max_scores(self, case, category_string, fixed_dict):
        """
        按顺序计算工具链中可能获得的最高分数
        
        Parameters:
        -----------
        case : dict
            包含案例所有信息的字典
        category_string : str
            工具调用链顺序的字符串表示，如"012356789"
        fixed_dict : dict
            已知的固定分数，这些分数在整个过程中保持不变，
            并且会限制所有相关变量的分数
        
        Returns:
        --------
        dict
            每个可能输出的最高分数
        """
        category_map = {
            '0': 'Anatomy Classifier',
            '1': 'Modality Classifier',
            '2': 'Organ Segmentor',
            '3': 'Anomaly Detector',
            '4': 'Disease Diagnoser',
            '5': 'Disease Inferencer',
            '6': 'Biomarker Quantifier',
            '7': 'Indicator Evaluator',
            '8': 'Report Generator',
            '9': 'Treatment Recommender'
        }
        
        category_list = [category_map[digit] for digit in category_string]
        max_score_dict = {}  # 不再直接复制fixed_dict

        for category in category_list:
            valid_tools = []
            for tool_name, info in self.toolinfo.items():
                if info['Category'] != category:
                    continue
                    
                # 检查工具是否满足案例要求
                is_valid = True
                
                # 检查解剖部位和模态
                if not ((info['Anatomy'] is None and info['Modality'] is None) or
                    (info['Anatomy'] == 'Universal' and info['Modality'] == 'Universal') or
                    (info['Anatomy'] == case['Anatomy'] and info['Modality'] == case['Modality'])):
                    continue
                
                # 检查器官支持
                if (info['Organs'] is not None and 
                    'OrganBiomarker' in case and 
                    'OrganObject' in case['OrganBiomarker']):
                    if not all(organ in info['Organs'] for organ in case['OrganBiomarker']['OrganObject']):
                        is_valid = False
                
                # 检查异常支持
                if (info['Anomalies'] is not None and 
                    'Anomaly' in case and 
                    'Symptom' in case['Anomaly']):
                    if not all(anomaly in info['Anomalies'] for anomaly in case['Anomaly']['Symptom']):
                        is_valid = False
                
                # 检查疾病支持
                if (info['Diseases'] is not None and 
                    'Disease' in case):
                    if not all(disease in info['Diseases'] for disease in case['Disease']):
                        is_valid = False
                
                if is_valid:
                    valid_tools.append(info)

            if not valid_tools:
                raise ValueError(f"No valid tool found for category {category}")

            # 计算当前类别的最高分数
            category_max_scores = {}
            
            # 对于Biomarker Quantifier和Indicator Evaluator，分别计算Organ和Anomaly的分数
            if category in ['Biomarker Quantifier', 'Indicator Evaluator']:
                # 分别处理Organ和Anomaly类型的工具
                organ_tools = [tool for tool in valid_tools if tool['type'] == 'Organ']
                anomaly_tools = [tool for tool in valid_tools if tool['type'] == 'Anomaly']
                
                # 处理Organ类型的工具
                for tool_info in organ_tools:
                    min_compulsory_score = 1.0
                    can_use_tool = True
                    
                    for input_item in tool_info['Compulsory Input']:
                        if input_item not in max_score_dict and input_item not in fixed_dict:
                            can_use_tool = False
                            print(f"Missing input: {input_item}")
                            break
                        # 使用fixed_dict中的值或max_score_dict中的值
                        input_score = fixed_dict.get(input_item, max_score_dict.get(input_item))
                        min_compulsory_score = min(min_compulsory_score, input_score)
                    
                    if not can_use_tool:
                        print(f"Cannot use tool {tool_info['Name']}")
                        continue

                    current_score = tool_info['lower_bound'] * min_compulsory_score
                    
                    if tool_info['Optional Input'] and tool_info['step'] > 0:
                        for input_item in tool_info['Optional Input']:
                            if input_item in fixed_dict or input_item in max_score_dict:
                                current_score += tool_info['step'] * (
                                    fixed_dict.get(input_item, max_score_dict.get(input_item))
                                )
                        current_score = min(current_score, tool_info['upper_bound'])

                    for output_item in tool_info['Output']:
                        if output_item in fixed_dict:  # 如果输出在fixed_dict中，使用fixed值
                            category_max_scores[output_item] = fixed_dict[output_item]
                        else:  # 否则使用计算值
                            category_max_scores[output_item] = max(
                                category_max_scores.get(output_item, 0),
                                current_score
                            )
                
                # 处理Anomaly类型的工具
                for tool_info in anomaly_tools:
                    min_compulsory_score = 1.0
                    can_use_tool = True
                    
                    for input_item in tool_info['Compulsory Input']:
                        if input_item not in max_score_dict and input_item not in fixed_dict:
                            can_use_tool = False
                            print(f"Missing input: {input_item}")
                            break
                        input_score = fixed_dict.get(input_item, max_score_dict.get(input_item))
                        min_compulsory_score = min(min_compulsory_score, input_score)
                    
                    if not can_use_tool:
                        print(f"Cannot use tool {tool_info['Name']}")
                        continue

                    current_score = tool_info['lower_bound'] * min_compulsory_score
                    
                    if tool_info['Optional Input'] and tool_info['step'] > 0:
                        for input_item in tool_info['Optional Input']:
                            if input_item in fixed_dict or input_item in max_score_dict:
                                current_score += tool_info['step'] * (
                                    fixed_dict.get(input_item, max_score_dict.get(input_item))
                                )
                        current_score = min(current_score, tool_info['upper_bound'])

                    for output_item in tool_info['Output']:
                        if output_item in fixed_dict:
                            category_max_scores[output_item] = fixed_dict[output_item]
                        else:
                            category_max_scores[output_item] = max(
                                category_max_scores.get(output_item, 0),
                                current_score
                            )
            
            else:
                # 处理其他类别的工具
                for tool_info in valid_tools:
                    min_compulsory_score = 1.0
                    can_use_tool = True
                    
                    for input_item in tool_info['Compulsory Input']:
                        if input_item not in max_score_dict and input_item not in fixed_dict:
                            can_use_tool = False
                            print(f"Missing input: {input_item}")
                            break
                        input_score = fixed_dict.get(input_item, max_score_dict.get(input_item))
                        min_compulsory_score = min(min_compulsory_score, input_score)
                    
                    if not can_use_tool:
                        print(f"Cannot use tool {tool_info['Name']}")
                        continue

                    current_score = tool_info['lower_bound'] * min_compulsory_score
                    
                    if tool_info['Optional Input'] and tool_info['step'] > 0:
                        for input_item in tool_info['Optional Input']:
                            if input_item in fixed_dict or input_item in max_score_dict:
                                current_score += tool_info['step'] * (
                                    fixed_dict.get(input_item, max_score_dict.get(input_item))
                                )
                        current_score = min(current_score, tool_info['upper_bound'])

                    for output_item in tool_info['Output']:
                        if output_item in fixed_dict:
                            category_max_scores[output_item] = fixed_dict[output_item]
                        else:
                            category_max_scores[output_item] = max(
                                category_max_scores.get(output_item, 0),
                                current_score
                            )

            # 更新总分数字典
            for item, score in category_max_scores.items():
                if item in fixed_dict:
                    max_score_dict[item] = fixed_dict[item]
                else:
                    max_score_dict[item] = score

        return max_score_dict
    
    def get_tool_by_name(self, tool_name):
        """根据工具名称获取工具信息"""
        if tool_name not in self.toolinfo:
            raise ValueError(f"Tool {tool_name} not found")
        return {
            'info': self.toolinfo[tool_name],
            'description': self.tooldes[tool_name],
            'function': self.toolbox[tool_name]
        }

    def print_tool_info(self, tool_name):
        """打印特定工具的详细信息"""
        if tool_name not in self.toolinfo:
            raise ValueError(f"Tool {tool_name} not found")
        print(f"=== Tool Information for {tool_name} ===")
        for key, value in self.toolinfo[tool_name].items():
            print(f"{key}: {value}")

    def print_tool_description(self, tool_name):
        """打印特定工具的描述信息"""
        if tool_name not in self.tooldes:
            raise ValueError(f"Tool {tool_name} not found")
        print(f"=== Tool Description for {tool_name} ===")
        for key, value in self.tooldes[tool_name].items():
            print(f"{key}: {value}")

    def print_all_tools_info(self):
        """打印所有工具的详细信息"""
        for tool_name in self.toolinfo:
            print("\n" + "="*50)
            self.print_tool_info(tool_name)

    def print_all_tools_description(self):
        """打印所有工具的描述信息"""
        for tool_name in self.tooldes:
            print("\n" + "="*50)
            self.print_tool_description(tool_name)

    def get_tools_by_category(self, category):
        """根据类别查找工具"""
        return [name for name, info in self.toolinfo.items() 
                if info['Category'] == category]

    def get_tools_by_anatomy(self, anatomy):
        """根据解剖部位查找工具"""
        return [name for name, info in self.toolinfo.items() 
                if info['Anatomy'] == anatomy]

    def get_tools_by_modality(self, modality):
        """根据成像模态查找工具"""
        return [name for name, info in self.toolinfo.items() 
                if info['Modality'] == modality]

    def get_tools_by_criteria(self, **kwargs):
        """根据多个条件查找工具
        
        Parameters:
        -----------
        **kwargs : dict
            可以包含 Category, Anatomy, Modality 等作为筛选条件
        
        Returns:
        --------
        list
            符合条件的工具名称列表
        """
        tools = set(self.toolinfo.keys())
        
        for key, value in kwargs.items():
            if value is not None:
                tools = {name for name in tools 
                        if self.toolinfo[name].get(key) == value}
        
        return list(tools)

    def print_tool_summary(self):
        """打印工具库摘要信息"""
        categories = set(info['Category'] for info in self.toolinfo.values())
        anatomies = set(info['Anatomy'] for info in self.toolinfo.values() 
                       if info['Anatomy'] is not None)
        modalities = set(info['Modality'] for info in self.toolinfo.values() 
                        if info['Modality'] is not None)
        
        print("=== Tool Library Summary ===")
        print(f"Total number of tools: {len(self.toolinfo)}")
        print("\nCategories:")
        for cat in categories:
            count = len([1 for info in self.toolinfo.values() 
                        if info['Category'] == cat])
            print(f"  - {cat}: {count} tools")
        
        print("\nAnatomies:")
        for anat in anatomies:
            count = len([1 for info in self.toolinfo.values() 
                        if info['Anatomy'] == anat])
            print(f"  - {anat}: {count} tools")
        
        print("\nModalities:")
        for mod in modalities:
            count = len([1 for info in self.toolinfo.values() 
                        if info['Modality'] == mod])
            print(f"  - {mod}: {count} tools")

    
if __name__ == "__main__":
    manager = ToolManager_IO()
    manager.build_tools()
    manager.print_tool_summary()
#     TOOLDES = tool_manager.tooldes
#     # print(TOOLDES)
#     TOOLBOX = tool_manager.toolbox
#     tool20_code = inspect.getsource(TOOLBOX['TOOL20'])
#     print(tool20_code)
    # specific_tools = manager.get_tools_by_criteria(
    #     Anatomy='Chest',
    #     Modality='CT',
    #     Category='Organ Segmentor'
    # )
    # print("Specific tools:", specific_tools)
    # manager.print_tool_description(specific_tools[0])
    # manager.print_all_tools_description()