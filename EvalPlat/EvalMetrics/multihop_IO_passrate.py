import re
import json
from Levenshtein import distance
from tqdm import tqdm
import os

def natural_sort_key(s):
    # 将字符串中的数字部分转换为整数，用于自然排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def detect_error(text):
    # Check for error message pattern 
    error_pattern = r"Error Type: (\w+)\nError Message: (.+?)\nTraceback:"
    error_match = re.search(error_pattern, text)
    
    if error_match:
        return False, error_match.group(1), error_match.group(2)
    return True, None, None

def extract_conclusion(text):
    # Look for conclusion pattern after "Final combination starts!"
    conclusion_pattern = r"Final combination starts!\s*Conclusion:(.*?)(?=\n\n|$)"
    match = re.search(conclusion_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_tool_chain_and_calls(text, tool_name_dict, TOOLINFO):
    # Extract tool chain
    tool_chain_pattern = r'Tool Chain: \[(.*?)\]'
    tool_chain_match = re.search(tool_chain_pattern, text)
    
    tool_names = []
    if tool_chain_match:
        tools_str = tool_chain_match.group(1)
        tool_matches = re.findall(r'\*(.*?)\*', tools_str)
        
        # Convert to normalized names using tool_name_dict
        for tool in tool_matches:
            if tool in tool_name_dict:
                tool_names.append(tool_name_dict[tool])

    # Extract Call Dicts and their tool categories
    call_dict_pattern = r"Call Dict: ({[^}]+})"
    call_dicts = re.findall(call_dict_pattern, text)
    
    tool_categories = []
    for call_dict_str in call_dicts:
        tool_match = re.search(r"'Tool': 'TOOL(\d+)'", call_dict_str)
        if tool_match:
            tool_number = "TOOL" + tool_match.group(1)
            if tool_number in TOOLINFO:
                category = TOOLINFO[tool_number]["Category"]
                tool_categories.append(category)

    return tool_names, tool_categories

def get_longest_common_subarray_length(arr1, arr2):
    m = len(arr1)
    n = len(arr2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if arr1[i-1] == arr2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
                
    return max_len

def process_case(text, tool_name_dict, TOOLINFO, tool_gts):
    success, error_type, error_msg = detect_error(text)
    
    tool_names, tool_categories = extract_tool_chain_and_calls(text, tool_name_dict, TOOLINFO)
    
    if not success:
        return {
            'success': False,
            'error_type': error_type,
            'error_message': error_msg,
            'distances': None,
            'tool_categories': tool_categories,
            'tool_length_ratio': len(tool_categories)/len(tool_gts) if tool_categories else 0
        }
    
    # Extract conclusion for successful cases
    conclusion = extract_conclusion(text)
    
    # Check if last tool matches
    last_tool_match = False
    if tool_categories and tool_gts:
        last_tool_match = (tool_categories[-1] == tool_gts[-1])
    
    # Calculate redundancy
    redundant_tools = [tool for tool in tool_categories if tool not in tool_gts]
    max_len = max(len(tool_categories), len(tool_gts))
    redundancy = len(redundant_tools) / max_len if max_len > 0 else 0
    
    # Calculate precision
    precision = get_longest_common_subarray_length(tool_gts, tool_categories) / len(tool_gts) if tool_gts else 0
    
    distances = {
        'gt_vs_categories': distance(tool_gts, tool_categories),
        'names_vs_categories': distance(tool_names, tool_categories),
        'gt_vs_names': distance(tool_gts, tool_names)
    }
    
    return {
        'success': True,
        'tool_names': tool_names,
        'tool_categories': tool_categories,
        'distances': distances,
        'last_tool_match': last_tool_match,
        'redundancy': redundancy,
        'precision': precision,
        'conclusion': conclusion
    }

def analyze_results(results):
    total_cases = len(results)
    successful_cases = [r for r in results if r['success']]
    failed_cases = [r for r in results if not r['success']]
    
    # Calculate pass rates
    passrate1 = len(successful_cases) / total_cases if total_cases > 0 else 0
    passrate2 = sum(1 for r in successful_cases if r['last_tool_match']) / total_cases if total_cases > 0 else 0
    
    # Calculate averages for successful cases
    avg_redundancy = sum(r['redundancy'] for r in successful_cases) / len(successful_cases) if successful_cases else 0
    avg_precision = sum(r['precision'] for r in successful_cases) / len(successful_cases) if successful_cases else 0
    
    # Calculate average tool length ratio for failed cases
    avg_failed_length_ratio = sum(r['tool_length_ratio'] for r in failed_cases) / len(failed_cases) if failed_cases else 0
    
    if successful_cases:
        distances = [case['distances'] for case in successful_cases]
        avg_distances = {
            'gt_vs_categories': sum(d['gt_vs_categories'] for d in distances) / len(distances),
            'names_vs_categories': sum(d['names_vs_categories'] for d in distances) / len(distances),
            'gt_vs_names': sum(d['gt_vs_names'] for d in distances) / len(distances)
        }
    else:
        avg_distances = None
    
    # Collect conclusions for successful cases
    conclusions = [r['conclusion'] for r in successful_cases if r['conclusion']]
    
    return {
        'total_cases': total_cases,
        'successful_cases': len(successful_cases),
        'failed_cases': len(failed_cases),
        'passrate1': passrate1,
        'passrate2': passrate2,
        'avg_redundancy': avg_redundancy,
        'avg_precision': avg_precision,
        'avg_failed_length_ratio': avg_failed_length_ratio,
        'average_distances': avg_distances,
        'error_types': [(case['error_type'], case['error_message']) for case in failed_cases],
        'conclusions': conclusions
    }

tool_name_dict = {
    "Anatomy Classification Tool": "Anatomy Classifier",
    "Modality Classification Tool": "Modality Classifier", 
    "Organ Segmentation Tool": "Organ Segmentor",
    "Anomaly Detection Tool": "Anomaly Detector",
    "Disease Diagnosis Tool": "Disease Diagnoser",
    "Disease Inference Tool": "Disease Inferencer",
    "Organ Biomarker Quantification Tool": "Organ Biomarker Quantifier",
    "Biomarker Quantification Tool": "Biomarker Quantifier",
    "Indicator Evaluator Tool": "Indicator Evaluator",
    "Report Generation Tool": "Report Generator",
    "Treatment Recommendation Tool": "Treatment Recommender"
}

map_dict = {
    "all": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Diagnoser", "Disease Inferencer", "Organ Biomarker Quantifier", 
            "Anomaly Biomarker Quantifier", "Organ Indicator Evaluator", "Anomaly Indicator Evaluator", "Indicator Evaluator", "Report Generator", "Treatment Recommender"],
    "012": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor"],
    "013": ["Anatomy Classifier", "Modality Classifier", "Anomaly Detector"],
    "014": ["Anatomy Classifier", "Modality Classifier", "Disease Diagnoser"],
    "0123": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector"],
    "0126": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Organ Biomarker Quantifier"],
    "0136": ["Anatomy Classifier", "Modality Classifier", "Anomaly Detector", "Anomaly Biomarker Quantifier"],
    "01235": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer"],
    "01348": ["Anatomy Classifier", "Modality Classifier", "Anomaly Detector", "Disease Diagnoser", "Report Generator"],
    "0123568": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer", "Biomarker Quantifier", "Biomarker Quantifier", "Report Generator"],    
    "01235678": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer", "Biomarker Quantifier", "Biomarker Quantifier", "Indicator Evaluator", "Report Generator"],
    "012356789": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer", "Biomarker Quantifier", "Biomarker Quantifier", "Indicator Evaluator", "Report Generator", "Treatment Recommender"],
}

def main():
    root_path = ".../path/to/your/Resultfolder"
    folder_names = os.listdir(f"{root_path}/logouts")
    folder_names.sort(key=natural_sort_key)
    
    results = []
    for folder_name in tqdm(folder_names, desc="Processing folders..."):
        logouts_folder_path = f"{root_path}/logouts/{folder_name}"
        file_name_list = os.listdir(logouts_folder_path)
        file_name_list.sort(key=natural_sort_key)
        
        for file_name in file_name_list:
            str_name = file_name.split(".")[0]
            file_path = f"{logouts_folder_path}/{file_name}"
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            info_path = f"{root_path}/info/{folder_name}/{str_name}.json"
            with open(info_path, 'r') as f:
                info_dict = json.load(f)
            
            TOOLINFO = info_dict["TOOLINFO"]
            case_result = process_case(text, tool_name_dict, TOOLINFO, map_dict[str_name])
            results.append(case_result)
    
    print("Sample Results (first 2):")
    # print(results[:2])
    print("\nAnalysis:")
    analysis = analyze_results(results)
    # print(analysis)
    print("Passrate 1:", analysis['passrate1'])
    print("Passrate 2:", analysis['passrate2'])
    print("Average Redundancy:", analysis['avg_redundancy'])
    print("Average Precision:", analysis['avg_precision'])
    print("Average Failed Length Ratio:", analysis['avg_failed_length_ratio'])
    print("Average Distances:")
    print(analysis['average_distances'])

if __name__ == "__main__":
    main()