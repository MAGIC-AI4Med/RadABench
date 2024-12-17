import re
import os
import json
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from nltk.tokenize import word_tokenize

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def extract_tool_chain_and_calls(text, TOOLINFO):
    # 提取Call Dict中的工具类别
    call_dict_pattern = r"Call Dict: ({[^}]+})"
    call_dicts = re.findall(call_dict_pattern, text)
    
    tool_categories = []
    tool_numbers = []
    
    for call_dict_str in call_dicts:
        tool_match = re.search(r"'Tool': 'TOOL(\d+)'", call_dict_str)
        if tool_match:
            tool_number = "TOOL" + tool_match.group(1)
            tool_numbers.append(tool_number)  # 存储工具编号
            if tool_number in TOOLINFO:
                category = TOOLINFO[tool_number]["Category"]
                tool_categories.append(category)
    
    return tool_categories, tool_numbers

def calculate_win_rate(tool_categories, chain_type, max_tool_dict):
    if chain_type not in max_tool_dict:
        return 0
    
    max_tools = max_tool_dict[chain_type]
    for i, tool in enumerate(reversed(max_tools)):
        # print(tool, tool_categories)
        # input()
        if tool in tool_categories:
            return (len(max_tools) - i) / len(max_tools)
    return 0

def check_milestone(tool_categories, chain_type, key_map):
    if chain_type not in key_map:
        return False
    target_category = key_map[chain_type]
    return target_category in tool_categories

def extract_conclusion(text):
    match = re.search(r'Conclusion:(.*?)(?=\n\n|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def calculate_metrics(reference, hypothesis):
    if not reference or not hypothesis:
        return 0, 0, 0
    
    rouge = Rouge()
    
    ref_tokens = set(word_tokenize(reference.lower()))
    hyp_tokens = set(word_tokenize(hypothesis.lower()))
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0, 0, 0
        
    common = ref_tokens.intersection(hyp_tokens)
    precision = len(common) / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算BLEU-1
    smoother = SmoothingFunction()
    ref_tokens = [word_tokenize(reference.lower())]
    hyp_tokens = word_tokenize(hypothesis.lower())
    weights = (1.0, 0, 0, 0)  # 只使用unigram
    bleu = sentence_bleu(ref_tokens, hyp_tokens, 
                        weights=weights,
                        smoothing_function=smoother.method1)
    
    try:
        rouge_scores = rouge.get_scores(hypothesis, reference)
        rouge_l = rouge_scores[0]['rouge-l']['f']
    except:
        rouge_l = 0
        
    return f1, bleu, rouge_l

def analyze_output(text, case, chain_type, max_tool_dict, key_map, TOOLINFO):
    has_error = 'Error' in text or 'error' in text
    
    # 使用新的方法提取工具类别
    tool_categories, tool_numbers = extract_tool_chain_and_calls(text, TOOLINFO)
    
    win_rate = calculate_win_rate(tool_numbers, chain_type, max_tool_dict)
    milestone = check_milestone(tool_categories, chain_type, key_map)
    
    if has_error:
        return {
            'has_error': True,
            'win_rate': win_rate,
            'milestone': milestone,
            'f1': 0,
            'bleu': 0,
            'rouge': 0
        }
    
    conclusion = extract_conclusion(text)
    reference = case['QA'].get(chain_type, {}).get('A', '')
    
    f1, bleu, rouge = calculate_metrics(reference, conclusion)
    
    return {
        'has_error': False,
        'win_rate': win_rate,
        'milestone': milestone,
        'f1': f1,
        'bleu': bleu,
        'rouge': rouge
    }

def main():
    max_tool_dict = {
        "012": ["TOOL3", "TOOL4", "TOOL5"],
        "013": ["TOOL5", "TOOL6", "TOOL7"],
        "014": ["TOOL9", "TOOL11", "TOOL7", "TOOL8", "TOOL10"],
        "0123": ["TOOL3", "TOOL4", "TOOL5"],
        "0126": ["TOOL10", "TOOL11"],
        "0136": ["TOOL11", "TOOL12"],
        "01235": ["TOOL7", "TOOL8", "TOOL10", "TOOL9", "TOOL11"],
        "01348": ["TOOL5", "TOOL6", "TOOL7"],
        "0123568": ["TOOL7", "TOOL8", "TOOL10", "TOOL9", "TOOL11"],
        "01235678": ["TOOL12", "TOOL14", "TOOL13", "TOOL15"],
        "012356789": ["TOOL12", "TOOL14", "TOOL13", "TOOL15"]
    }
    
    key_map = {
        "012": 'Organ Segmentor',
        "013": 'Anomaly Detector',
        "014": 'Disease Diagnoser',
        "0123": 'Organ Segmentor',
        "01235": 'Disease Inferencer',
        "0126": 'Biomarker Quantifier',
        "0136": 'Biomarker Quantifier',
        "01348": 'Anomaly Detector',
        "0123568": 'Disease Diagnoser',
        "01235678": 'Indicator Evaluator',
        "012356789": 'Indicator Evaluator'
    }

    # 加载cases
    with open(".../path/to/your/Datafolder/combined_casesV3_qa_Subset.json", 'r') as f:
        combined_cases = json.load(f)
    case_list = list(combined_cases.values())

    total_stats = defaultdict(lambda: {
        'count': 0,
        'win_rate': 0,
        'milestone': 0,
        'f1': 0,
        'bleu': 0,
        'rouge': 0,
        'valid_count': 0
    })

    root_path = ".../path/to/your/Resultfolder"
    folder_names = os.listdir(f"{root_path}/logouts")
    folder_names.sort(key=natural_sort_key)
    
    for folder_name in tqdm(folder_names):
        case_idx = int(folder_name)
        if case_idx >= len(case_list):
            continue
            
        current_case = case_list[case_idx]['case']
        folder_path = f"{root_path}/logouts/{folder_name}"
        file_names = os.listdir(folder_path)
        
        for file_name in file_names:
            chain_type = file_name.split('.')[0]
            str_name = file_name.split(".")[0]
            # print(chain_type, str_name)
            # input()
            
            # 加载TOOLINFO
            info_path = f"{root_path}/info/{folder_name}/{str_name}.json"
            with open(info_path, 'r') as f:
                info_dict = json.load(f)
            TOOLINFO = info_dict["TOOLINFO"]
            
            with open(f"{folder_path}/{file_name}", 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = analyze_output(text, case_list[case_idx], chain_type, 
                                  max_tool_dict, key_map, TOOLINFO)
            
            stats = total_stats[chain_type]
            stats['count'] += 1
            stats['win_rate'] += result['win_rate']
            stats['milestone'] += int(result['milestone'])
            
            if not result['has_error']:
                stats['valid_count'] += 1
                stats['f1'] += result['f1']
                stats['bleu'] += result['bleu']
                stats['rouge'] += result['rouge']

    # 打印结果
    # for chain_type, stats in total_stats.items():
    #     count = stats['count']
    #     valid_count = stats['valid_count']
        
    #     print(f"\nChain Type: {chain_type}")
    #     print(f"Total cases: {count}")
    #     print(f"Win Rate: {stats['win_rate']/count:.4f}")
    #     print(f"Milestone Rate: {stats['milestone']/count:.4f}")
    #     if valid_count > 0:
    #         print(f"F1: {stats['f1']/valid_count:.4f}")
    #         print(f"BLEU: {stats['bleu']/valid_count:.4f}")
    #         print(f"ROUGE: {stats['rouge']/valid_count:.4f}")
    #     else:
    #         print("No valid cases for F1/BLEU/ROUGE calculation")
    # 打印结果
    total_count = 0
    total_valid_count = 0
    total_win_rate = 0
    total_milestone = 0
    total_f1 = 0
    total_bleu = 0
    total_rouge = 0

    # 累加所有统计数据
    for stats in total_stats.values():
        count = stats['count']
        valid_count = stats['valid_count']
        
        total_count += count
        total_valid_count += valid_count
        total_win_rate += stats['win_rate']
        total_milestone += stats['milestone']
        total_f1 += stats['f1']
        total_bleu += stats['bleu']
        total_rouge += stats['rouge']

    # 计算总体平均值
    print(f"Total cases: {total_count}")
    print(f"Average Win Rate: {total_win_rate/total_count:.4f}")
    print(f"Average Milestone Rate: {total_milestone/total_count:.4f}")
    if total_valid_count > 0:
        print(f"Average F1: {total_f1/total_valid_count:.4f}")
        print(f"Average BLEU: {total_bleu/total_valid_count:.4f}")
        print(f"Average ROUGE: {total_rouge/total_valid_count:.4f}")
    else:
        print("No valid cases for F1/BLEU/ROUGE calculation")

if __name__ == "__main__":
    main()