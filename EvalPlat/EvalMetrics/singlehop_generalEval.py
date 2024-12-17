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
    "0123568": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer", "Organ Biomarker Quantifier", "Anomaly Biomarker Quantifier", "Report Generator"],    
    "01235678": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer", "Organ Biomarker Quantifier", "Anomaly Biomarker Quantifier", "Indicator Evaluator", "Report Generator"],
    "012356789": ["Anatomy Classifier", "Modality Classifier", "Organ Segmentor", "Anomaly Detector", "Disease Inferencer", "Organ Biomarker Quantifier", "Anomaly Biomarker Quantifier", "Indicator Evaluator", "Report Generator", "Treatment Recommender"],
}

# file_path = "/remote-home/qiaoyuzheng/MedAgents/AgentCoreV3/GPT4/General_backup/logouts/0/012.txt"  # Replace with your file path
# result = extract_tools(file_path)
# print(result)

root_path = ".../path/to/your/Resultfolder"

edit_distance_list = []
for i in tqdm(range(len(os.listdir(f"{root_path}/logouts"))), desc="Processing..."):
    logouts_folder_path = f"{root_path}/logouts/{i}"
    file_name_list = os.listdir(logouts_folder_path)
    file_name_list.sort(key=natural_sort_key)
    for file_name in file_name_list:
        str_name = file_name.split(".")[0]
        file_path = f"{logouts_folder_path}/{file_name}"
        tools = extract_tools(file_path)
        info_path = f"{root_path}/info/{i}/{str_name}.json"
        with open(info_path, 'r') as f:
            info_dict = json.load(f)
        tool_category_list = []
        for tool in tools:
            if tool not in info_dict["TOOLINFO"]:
                continue
            tool_category = info_dict["TOOLINFO"][tool]["Category"]
            if tool_category == 'Biomarker Quantifier':
                tool_category = f"{info_dict['TOOLINFO'][tool]['type']} Biomarker Quantifier"
            tool_category_list.append(tool_category)
        tool_category_list = sort_by_reference(tool_category_list, map_dict[str_name])
        tool_category_reference = map_dict[str_name]
        #calculate the levinshtein distance between tool_category_list and tool_category_reference
        edit_distance = distance(tool_category_list, tool_category_reference)
        edit_distance_list.append(edit_distance)


print(edit_distance_list)
print(sum(edit_distance_list)/len(edit_distance_list))