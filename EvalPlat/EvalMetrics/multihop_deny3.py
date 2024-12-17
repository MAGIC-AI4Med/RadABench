# deny_map = {"012": 'Organ Segmentor', "013": 'Anomaly Detector', "014": 'Disease Diagnoser', "0123": 'Organ Segmentor', "01235": 'Disease Diagnoser', "0126": 'Biomarker Quantifier',
#             "0136": 'Biomarker Quantifier', "01348": 'Anomaly Detector', "0123568": 'Disease Diagnoser', "01235678": 'Indicator Evaluator', "012356789": 'Indicator Evaluator'}
import re
import os
import json
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def check_deny_or_invalid(text):
    has_deny = 'Deny Call Dict' in text
    has_invalid = '<INVALID>' in text and '</INVALID>' in text
    return has_deny or has_invalid

def check_capability_match(text, case_id, deny_map):
    if case_id not in deny_map:
        return False
        
    expected_tool = deny_map[case_id]
    
    # 处理第一种情况 (Deny Call Dict with NoCall)
    if 'Deny Call Dict' in text:
        # 检查NoCall标签内的内容
        nocall_match = re.search(r'<NoCall>(.*?)</NoCall>', text, re.DOTALL)
        if nocall_match:
            nocall_content = nocall_match.group(1)
            has_insufficient = 'InsufficientCapability' in nocall_content
            has_expected_category = expected_tool in nocall_content
            if has_insufficient and has_expected_category:
                return True
        
        # 检查Deny Call Dict的内容
        deny_dict_match = re.search(r"Deny Call Dict: ({[^}]+})", text)
        if deny_dict_match:
            deny_dict_str = deny_dict_match.group(1)
            has_insufficient = 'InsufficientCapability' in deny_dict_str
            has_expected_category = expected_tool in deny_dict_str
            return has_insufficient and has_expected_category
            
    # 处理第二种情况 (<INVALID>)
    if '<INVALID>' in text and '</INVALID>' in text:
        invalid_content = re.search(r'<INVALID>([^<]+)</INVALID>', text)
        if invalid_content:
            invalid_text = invalid_content.group(1)
            has_capability = 'capability' in invalid_text.lower()
            has_expected_tool = expected_tool in invalid_text
            return has_capability and has_expected_tool
            
    return False

def get_case_id(text):
    match = re.search(r'Case \d+, Question \d+, (\d+):', text)
    if match:
        return match.group(1)
    return None

def process_file(text, deny_map):
    case_id = get_case_id(text)
    # print(case_id)
    # input()
    if not case_id:
        return None
        
    return {
        'has_deny_or_invalid': check_deny_or_invalid(text),
        'capability_match': check_capability_match(text, case_id, deny_map),
        'case_id': case_id
    }

def main():
    deny_map = {
        "012": 'Organ Segmentor', 
        "013": 'Anomaly Detector', 
        "014": 'Disease Diagnoser', 
        "0123": 'Organ Segmentor',
        "01235": 'Disease Diagnoser',
        "0126": 'Biomarker Quantifier',
        "0136": 'Biomarker Quantifier',
        "01348": 'Anomaly Detector',
        "0123568": 'Disease Diagnoser',
        "01235678": 'Indicator Evaluator',
        "012356789": 'Indicator Evaluator'
    }

    # with open("/remote-home/qiaoyuzheng/MedAgents/AgentCoreV3/CASE/combined_casesV3_qa_Subset.json", 'r') as f:
    #     combined_cases = json.load(f)
    # case_list = list(combined_cases.values())

    root_path = ".../path/to/your/Resultfolder"
    folder_names = os.listdir(f"{root_path}/logouts")
    folder_names.sort(key=natural_sort_key)
    
    total_cases = 0
    deny_invalid_pass = 0
    capability_pass = 0
    
    for folder_name in tqdm(folder_names):
        folder_path = f"{root_path}/logouts/{folder_name}"
        file_names = os.listdir(folder_path)
        file_names.sort(key=natural_sort_key)

        for file_name in file_names:
            file_path = f"{folder_path}/{file_name}"
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = process_file(text, deny_map)
            if not result:
                continue
                
            total_cases += 1
            if result['has_deny_or_invalid']:
                deny_invalid_pass += 1
            if result['capability_match']:
                capability_pass += 1
    
    deny_invalid_rate = deny_invalid_pass / total_cases if total_cases > 0 else 0
    capability_rate = capability_pass / total_cases if total_cases > 0 else 0
    
    print(f"Total cases analyzed: {total_cases}")
    print(f"Deny/Invalid check pass rate: {deny_invalid_rate:.2%}")
    print(f"Capability check pass rate: {capability_rate:.2%}")

if __name__ == "__main__":
    main()