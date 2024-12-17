import re
import os
from tqdm import tqdm
import json
from Levenshtein import distance


def natural_sort_key(s):
    # 将字符串中的数字部分转换为整数，用于自然排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def sort_by_reference(sublist, reference_list):
    # 创建一个字典，存储参考列表中每个元素的索引
    reference_dict = {item: index for index, item in enumerate(reference_list)}
    
    # 根据参考列表中的索引对子列表进行排序
    sorted_sublist = sorted(sublist, key=lambda x: reference_dict.get(x, len(reference_list)))
    return sorted_sublist

def extract_tools(file_path):
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Find content between <Decomposition> tags
        decomp_pattern = r'<Decomposition>(.*?)</Decomposition>'
        decomp_content = re.search(decomp_pattern, text, re.DOTALL)
        
        if decomp_content:
            # Find all tool tags and extract their content
            tool_pattern = r'<Tool>\s*(.*?)\s*</Tool>'
            tools = re.findall(tool_pattern, decomp_content.group(1))
            
            # Remove whitespace and store in list
            tools = [tool.strip() for tool in tools]
            
            return tools
        return []
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

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

tool_name_dict = {"Anatomy Classification Tool": "Anatomy Classifier", "Modality Classification Tool": "Modality Classifier", "Organ Segmentation Tool": "Organ Segmentor",
                  "Anomaly Detection Tool": "Anomaly Detector", "Disease Diagnosis Tool": "Disease Diagnoser", "Disease Inference Tool": "Disease Inferencer",
                    "Organ Biomarker Quantification Tool": "Organ Biomarker Quantifier", "Biomarker Quantification Tool": "Biomarker Quantifier",
                    "Indicator Evaluator Tool": "Indicator Evaluator", "Report Generation Tool": "Report Generator", "Treatment Recommendation Tool": "Treatment Recommender"}


def extract_tool_chain_and_calls(text, tool_name_dict, TOOLINFO):
    # Extract tool chain
    tool_chain_pattern = r'Tool Chain: \[(.*?)\]'
    tool_chain_match = re.search(tool_chain_pattern, text)
    
    tool_names = []
    if tool_chain_match:
        # Extract content between * characters
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
        # Extract TOOL number
        tool_match = re.search(r"'Tool': 'TOOL(\d+)'", call_dict_str)
        if tool_match:
            tool_number = "TOOL" + tool_match.group(1)
            # Get category from TOOLINFO
            if tool_number in TOOLINFO:
                category = TOOLINFO[tool_number]["Category"]
                tool_categories.append(category)

    return tool_names, tool_categories

root_path = ".../path/to/your/Resultfolder"
edit_distance_list1 = []
edit_distance_list2 = []
edit_distance_list3 = []
for i in tqdm(range(2,len(os.listdir(f"{root_path}/logouts"))), desc="Processing..."):
    logouts_folder_path = f"{root_path}/logouts/{i}"
    file_name_list = os.listdir(logouts_folder_path)
    file_name_list.sort(key=natural_sort_key)
    for file_name in file_name_list:
        str_name = file_name.split(".")[0]
        file_path = f"{logouts_folder_path}/{file_name}"
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        info_path = f"{root_path}/info/{i}/{str_name}.json"
        with open(info_path, 'r') as f:
            info_dict = json.load(f)
        TOOLINFO = info_dict["TOOLINFO"]
        tool_names, tool_categories = extract_tool_chain_and_calls(text, tool_name_dict, TOOLINFO)
        tool_gts = map_dict[str_name]
        edit_distance_list1.append(distance(tool_gts, tool_categories))
        edit_distance_list2.append(distance(tool_names, tool_categories))
        edit_distance_list3.append(distance(tool_gts, tool_names))

print("Edit distance between ground truth and predicted tool categories:", edit_distance_list1)
print("Edit distance between predicted tool names and predicted tool categories:", edit_distance_list2)
print("Edit distance between ground truth and predicted tool names:", edit_distance_list3)

print(len(edit_distance_list1), sum(edit_distance_list1) / len(edit_distance_list1))
print(len(edit_distance_list2), sum(edit_distance_list2) / len(edit_distance_list2))
print(len(edit_distance_list3), sum(edit_distance_list3) / len(edit_distance_list3))
