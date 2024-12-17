import re
import os
import json
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def analyze_nocall(text, target_category):
    # Check if Deny Call Dict exists
    has_deny = 'Deny Call Dict:' in text
    
    if not has_deny:
        return {
            'has_deny': False,
            'matches_category': False,
            'fields': None
        }
    
    # Extract fields using regex patterns
    purpose = re.search(r"'Purpose': '([^']+)'", text)
    category = re.search(r"'Category': '([^']+)'", text) 
    anatomy = re.search(r"'Anatomy': '([^']+)'", text)
    modality = re.search(r"'Modality': '([^']+)'", text)
    ability = re.search(r"'Ability': '([^']+)'", text)
    
    # Create fields dictionary with extracted values
    fields = {
        'Purpose': purpose.group(1) if purpose else None,
        'Category': category.group(1) if category else None,
        'Anatomy': anatomy.group(1) if anatomy else None,
        'Modality': modality.group(1) if modality else None,
        'Ability': ability.group(1) if ability else None
    }
    
    # Check if Category matches target
    category_match = False
    if fields['Category']:
        if target_category in fields['Category']:
            category_match = True
            
    return {
        'has_deny': True,
        'matches_category': category_match, 
        'fields': fields
    }

def main():
    deny_map = {"012": 'Anatomy Classifier', "013": 'Modality Classifier', "014": 'Disease Diagnoser', "0123": 'Organ Segmentor', "01235": 'Anomaly Detector', "0126": 'Biomarker Quantifier',
                 "0136": 'Biomarker Quantifier', "01348": 'Anomaly Detector', "0123568": 'Disease Diagnoser', "01235678": 'Report Generator', "012356789": 'Treatment Recommender'}

    root_path = ".../path/to/your/Resultfolder/logouts"  # Modify path as needed
    total_cases = 0
    total_denies = 0
    total_category_matches = 0
    
    folder_names = os.listdir(root_path)
    folder_names.sort(key=natural_sort_key)
    
    for folder_name in tqdm(folder_names):
        folder_path = f"{root_path}/{folder_name}"
        file_names = os.listdir(folder_path)
        file_names.sort(key=natural_sort_key)
        
        for file_name in file_names:
            str_name = file_name.split('.')[0]
            if str_name not in deny_map:
                continue
                
            file_path = f"{folder_path}/{file_name}"
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            target_category = deny_map[str_name]
            result = analyze_nocall(text, target_category)
            
            total_cases += 1
            if result['has_deny']:
                total_denies += 1
            if result['matches_category']:
                total_category_matches += 1
    
    deny_rate = total_denies / total_cases if total_cases > 0 else 0
    category_match_rate = total_category_matches / total_cases if total_cases > 0 else 0
    
    print(f"Total cases analyzed: {total_cases}")
    print(f"Deny rate: {deny_rate:.2%}")
    print(f"Category match rate: {category_match_rate:.2%}")

if __name__ == "__main__":
    main()