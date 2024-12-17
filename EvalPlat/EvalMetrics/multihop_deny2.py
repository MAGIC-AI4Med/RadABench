import re
import os
import json
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def analyze_nocall(text, case):
    # Check if Deny Call Dict exists  
    has_deny = 'Deny Call Dict:' in text
    
    if not has_deny:
        return {
            'has_deny': False,
            'matches_anatomy': False,
            'matches_modality': False,
            'fields': None
        }

    # Extract fields using regex patterns
    purpose = re.search(r"'Purpose': '([^']+)'", text)
    category = re.search(r"'Category': '([^']+)'", text)
    anatomy = re.search(r"'Anatomy': '([^']+)'", text)
    modality = re.search(r"'Modality': '([^']+)'", text)
    ability = re.search(r"'Ability': '([^']+)'", text)
    
    fields = {
        'Purpose': purpose.group(1) if purpose else None,
        'Category': category.group(1) if category else None, 
        'Anatomy': anatomy.group(1) if anatomy else None,
        'Modality': modality.group(1) if modality else None,
        'Ability': ability.group(1) if ability else None
    }

    # Check if Anatomy and Modality match
    anatomy_match = False
    modality_match = False
    
    if fields['Anatomy'] and 'Anatomy' in case:
        if fields['Anatomy'] == case['Anatomy'] or fields['Anatomy'] == 'Universal':
            anatomy_match = True
            
    if fields['Modality'] and 'Modality' in case:
        if fields['Modality'] == case['Modality'] or fields['Modality'] == 'Universal':
            modality_match = True

    return {
        'has_deny': True,
        'matches_anatomy': anatomy_match,
        'matches_modality': modality_match,
        'fields': fields
    }

def main():
    # Load cases
    with open(".../path/to/your/Datafolder/combined_casesV3_qa_Subset.json", 'r') as f:
        combined_cases = json.load(f)
    case_list = list(combined_cases.values())

    root_path = ".../path/to/your/Resultfolder"
    folder_names = os.listdir(f"{root_path}/logouts")
    folder_names.sort(key=natural_sort_key)
    
    total_cases = 0
    total_denies = 0
    total_anatomy_matches = 0
    total_modality_matches = 0
    total_both_matches = 0
    
    for folder_name in tqdm(folder_names):
        # Convert folder name to case index
        case_idx = int(folder_name)
        if case_idx >= len(case_list):
            continue
            
        current_case = case_list[case_idx]['case']
        # print(current_case)
        # input()
        folder_path = f"{root_path}/logouts/{folder_name}"
        file_names = os.listdir(folder_path)
        file_names.sort(key=natural_sort_key)
        
        for file_name in file_names:
            file_path = f"{folder_path}/{file_name}"
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = analyze_nocall(text, current_case)
            
            total_cases += 1
            if result['has_deny']:
                total_denies += 1
            if result['matches_anatomy']:
                total_anatomy_matches += 1
            if result['matches_modality']:
                total_modality_matches += 1
            if result['matches_anatomy'] and result['matches_modality']:
                total_both_matches += 1
    
    deny_rate = total_denies / total_cases if total_cases > 0 else 0
    both_match_rate = total_both_matches / total_cases if total_cases > 0 else 0
    
    print(f"Total cases analyzed: {total_cases}")
    print(f"Deny rate: {deny_rate:.2%}")
    print(f"Both Anatomy and Modality match rate: {both_match_rate:.2%}")
    print(f"Anatomy match rate: {total_anatomy_matches/total_cases:.2%}")
    print(f"Modality match rate: {total_modality_matches/total_cases:.2%}")

if __name__ == "__main__":
    main()