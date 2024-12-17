import requests
from requests.exceptions import ProxyError, ConnectionError
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import re
import traceback
import random
import ast
from openai import OpenAI
import inspect
from frozendict import frozendict
import inspect
import pickle
import dill

initial_prompt = """
    # Medical Image Analysis Assistant

    ## Task Overview
    You are a radiological agent analyzing medical images. For each query, you will receive:
    1. A medical imaging examination (Image) of a patient (assume already provided)
    2. Known patient Information including demographics, medical history and main complaints.

    Your task involves three sequential parts:

    1. Problem Decomposition (Part 1)
    - Identify available information
    - Break down the question into sequential steps

    2. Sequential Tool Application (Part 2)  
    - Execute one tool at a time
    - Record each tool's output
    - Continue until sufficient information is gathered

    3. Solution Synthesis (Part 3)
    - Integrate all results
    - Generate final answer

    ## Available Information Categories
    The following categories must be used exactly as written:

    ['$Information$', '$Anatomy$', '$Modality$', '$Disease$', '$OrganObject$', '$OrganDim$', '$OrganQuant$', '$AnomalyObject$', '$AnomalyDim$', '$AnomalyQuant$', '$IndicatorName$', '$IndicatorValue$', '$Report$', '$Treatment$']

    Where:
    - $Information$: Patient demographics (e.g., "45-year-old male", "BMI: 24", "history of diabetes")
    - $Anatomy$: Body part (e.g., "chest", "abdomen", "brain")
    - $Modality$: Imaging technique (e.g., "X-ray", "CT", "MRI")
    - $Disease$: Medical condition (e.g., "pneumonia", "cancer", "fracture") 
    - $OrganObject$: Organ to measure (e.g., "liver", "heart")
    - $OrganDim$: Organ measurement type (e.g., "number", "length", "size", "volume", "angle", "density", "intensity", "texture")
    - $OrganQuant$: Organ measurement value (e.g., "5cm", "120ml")
    - $AnomalyObject$: Abnormality to measure (e.g., "tumor", "fracture")
    - $AnomalyDim$: Abnormality measurement type (e.g., "number", "length", "size", "volume", "angle", "density", "intensity", "texture")
    - $AnomalyQuant$: Abnormality measurement value (e.g., "2cm", "5ml")
    - $IndicatorName$: Clinical indicator name
    - $IndicatorValue$: Clinical indicator value
    - $Report$: Medical report content
    - $Treatment$: Treatment recommendations

    ## Available Tool Categories
    Tool categories must be used exactly as written:

    [*Anatomy Classification Tool*, *Modality Classification Tool*, *Organ Segmentation Tool*, *Anomaly Detection Tool*, *Disease Diagnosis Tool*, *Disease Inference Tool*, *Organ Biomarker Quantification Tool*, *Anomaly Biomarker Quantification Tool*, *Indicator Evaluation Tool*, *Report Generation Tool*, *Treatment Recommendation Tool*]

    ## Response Format for Part 1
    For each query, respond ONLY with:

    Known Info: [list any categories explicitly mentioned in the query]
    Tool Chain: [list required tools connected by ->]

    ## Examples

    Query 1: "For a straightforward approach to diagnosing the patient's condition based on her symptoms and the image, what diseases can be directly identified?"
    Response:
    Known Info: []
    Tool Chain: [*Anatomy Classification Tool* -> *Modality Classification Tool* -> *Disease Diagnosis Tool*]

    Query 2: "This 45-year-old male's chest CT shows a 2cm nodule in the right lung. Can you give a report?"
    Response:
    Known Info: ['$Information$', '$Anatomy$', '$Modality$', '$AnomalyObject$', '$AnomalyDim$', '$AnomalyQuant$']
    Tool Chain: [*Organ Segmentation Tool* -> *Anomaly Detection Tool* -> *Disease Inference Tool* -> *Report Generation Tool*] (because some information is provided, so *Anatomy Classification Tool*, *Modality Classification Tool*, *Anomaly Biomarker Quantification Tool* are optimized.)

    ## Important Rules
    1. Assume the medical image is already provided
    2. Use exact item category names with $$ as listed (e.g., '$Anatomy$')
    3. Use exact tool category names with ** as listed (e.g., '*Anatomy Classification Tool*')
    4. Only respond with Part 1 analysis - Parts 2 & 3 will be addressed in subsequent interactions
    5. Include only the categories explicitly mentioned in the query
    6. Connect tools using -> symbol

    Please wait for my query. When provided, analyze it following the format shown in the examples above.
"""

# initial_prompt = """
#     Imagine a clinical scenario where you have:

#     1. A medical imaging examination $Image$ of a patient (Assume you have already received the image)
#     2. Known patient $Information$ including height, weight, age, gender, medical history and chief complaints.

#     As a radiological agent, your task is to solve a specific question/task related to this scenario using Chain of Thought (COT) reasoning (will be provided later). Please complete the following three parts:

#     Part 1. General promblem decomposition. 
#     - Identify all available information from the question/task description.
#     - Break down the given question/task into sequential steps that can be solved using different tools.
    
#     Part 2. Sequential Tool Utilization
#     - Execute each step by calling the appropriate tool
#     - Only one tool can be used per round
#     - Record the output/result from each tool
#     - Continue this process until you have sufficient information to solve the problem
    
#     Part 3. Solution Synthesis
#     - Analyze and integrate all results obtained from the tools and generate a final answer corresponding to the question/task.


#     Now please begin with #Part 1#. General problem decomposition.

#     All possible item categories are:
#     ['$Information$', '$Anatomy$', '$Modality$', '$Disease$', '$OrganObject$', '$OrganDim$', '$OrganQuant$', '$AnomalyObject$', '$AnomalyDim$', '$AnomalyQuant$', '$IndicatorName$', '$IndicatorValue$', '$Report$', '$Treatment$']
#     Where:
#     $Information$: Specific patient information (e.g., "45-year-old male", "BMI: 24", "history of diabetes")
#     $Anatomy$: Specific body part mentioned (e.g., "chest", "abdomen", "brain")
#     $Modality$: Specific imaging technique mentioned (e.g., "X-ray", "CT", "MRI")
#     $Disease$: Specific disease or condition mentioned (e.g., "pneumonia", "cancer", "fracture")
#     $OrganObject$: Specific organ-level object to be measured (e.g., "liver", "heart")
#     $OrganDim$: Specific dimension type (e.g., "number", "length", "size", "volume", "angle", "density", "intensity" or "texture")
#     $OrganQuant$: Specific numerical measurement value (e.g., "5cm", "120ml")
#     $AnomalyObject$: Specific anomaly-level object to be measured (e.g., "tumor", "fracture")
#     $AnomalyDim$: Specific dimension type (e.g., "number", "length", "size", "volume", "angle", "density", "intensity" or "texture")
#     $AnomalyQuant$: Similar to organ measurements but for anomalies (e.g., "2cm", "5ml")
#     $IndicatorName$, $IndicatorValue$: Specific indicator and its corresponding value (e.g., "ASPECT score: 8")
#     $Report$: Specific medical report content
#     $Treatment$: Specific treatment recommendations
    
#     All possible tool categories are:
#     [*Anatomy Classification Tool*, *Modality Classification Tool*, *Organ Segmentation Tool*, *Anomaly Detection Tool*, *Disease Diagnosis Tool*, *Disease Inference Tool*, *Organ Biomarker Quantification Tool*, *Anomaly Biomarker Quantification Tool*, *Indicator Evaluation Tool*, *Report Generation Tool*, *Treatment Recommendation Tool*]. Each tool has specific capabilities and inputs/outputs. You can refer to the tool descriptions for more details. 

#     #You should analyze the question and identify what specific information is already explicitly provided in the question.
#     #You should descide the tool chain using tools from the possible tool category list and use -> to connect the tools.
#     please combine the specific information and the tool chain to finish the first part of the task.
    
#     Examples:    

#     Question: "For a straightforward approach to diagnosing the patient's condition based on her symptoms and the image, what diseases can be directly identified?"
#     Response: 
#     Known Info: [] (because anatomy and modality are specifically mentioned)
#     Tool Chain: [*Organ Segmentation Tool* -> *Anomaly Detection Tool* -> *Disease Diagnosis Tool*]

#     Question: "This 45-year-old male's chest CT shows a 2cm nodule in the right lung. Can you give a report?"
#     Response: 
#     Known Info: ['$Information$', '$Anatomy$', '$Modality$', '$AnomalyObject$', '$AnomalyDim$' '$AnomalyQuant$'] (because specific age/gender, specific anatomy, specific modality, specific abnormality and its size are mentioned)
#     Tool Chain: [*Organ Segmentation Tool* -> *Anomaly Detection Tool* -> *Disease Inference Tool* -> *Report Generation Tool*]

#     Now please wait for my query, as long as you recieve the query, please response as the examples shows. (Note that you should assume the image is already provided). 
#     Specially pay attention that:
#     1. The Known Info list and Tool Chain list should be a subset of their corresponding categories above, using exactly the same format (e.g., '$Anatomy$' / *Anatomy Classification Tool*). Do not include any other text in your response.
#     2. Only answer Part 1. The Part 2 and 3 should be answered in the later conversations.
#     """

def prepare_sigle_hop_prompt(TOOLDES):
    prompt = f"""
    Imagine a clinical scenario: I will provide a radiological image and a question about that image. Your task is to act as a Radiological Agent by decomposing the question's task into a series of thought chains. You will answer the question step by step by calling different tools. However, the number of tools you can use, their capabilities, and their inputs/outputs are limited. I will provide you with a toolbox containing the following tools:
    
    {TOOLDES}

    Your response should include a series of steps. Each step must include:

    The purpose of the step (e.g., determining modality, identifying body part, detecting abnormalities, etc.)
    The tool from the toolbox that is called for this step
    The input for this step
    The output of this step
    Please use the following template to standardize your output:

    <Decomposition>

        <Call1>
            <Purpose> Brief, clear statement of the purpose of this tool call. </Purpose>
            <Tool> Name of the tool called. (named with the number as described in tool description) </Tool>
            <Input> List all the input to this tool, using $$ as the special mark </Input>
            <Output> List all the output of this tool, using $$ as the special mark </Output>
        </Call1>

        <Call2>
        ...
        </Call2>

        ...

    </Decomposition>

    Important notes:

    The initial input is only [$Information$, $Image$].
    The output of each tool call will be stored and can be used as input for subsequent tool calls.
    Assume that I have already provided you with the radiological image.
    I will now give you a question. Please generate the output as requested.
    """
    return prompt

def select_elements(target_str, input_list, upper_limit):
    # 找到目标字符串的位置
    pos = input_list.index(target_str)
    
    # 确定所在区间
    interval_start = (pos // 100) * 100
    interval_end = interval_start + 100
    
    # 获取该区间内除了目标字符串外的所有元素
    interval_elements = input_list[interval_start:interval_end]
    interval_elements.remove(target_str)
    
    # 随机选择元素个数(不超过上限)
    select_count = min(len(interval_elements), upper_limit)
    result = random.sample(interval_elements, select_count)
    
    return result

def extract_call_content(text):
    # 使用正则表达式提取<Call>标签之间的内容
    # print(text)
    call_pattern = re.compile(r'<(Call|EndCall)>(.*?)</\1>', re.DOTALL)
    call_content = call_pattern.search(text)
    
    tag_type = call_content.group(1)
    content = call_content.group(2)
    
    flag = True if tag_type == 'Call' else False

    # 提取Purpose, Tool, 和 Input
    purpose_pattern = re.compile(r'<Purpose>(.*?)</Purpose>', re.DOTALL)
    tool_pattern = re.compile(r'<Tool>(.*?)</Tool>', re.DOTALL)
    input_pattern = re.compile(r'<Input>(.*?)</Input>', re.DOTALL)
    
    purpose = purpose_pattern.search(content)
    tool = tool_pattern.search(content)
    input_value = input_pattern.search(content)
    # print(input_value.group(1).strip())
    # print(type(input_value.group(1).strip()))
    input_list = []
    for item in ['Image', 'Information', 'Anatomy', 'Modality', 'OrganMask', 'AnomalyMask', 'Disease', 'OrganObject', 'OrganDim', 'OrganQuant', 'AnomalyObject', 'AnomalyDim', 'AnomalyQuant', 'IndicatorName', 'IndicatorValue', 'Report', 'Treatment']:
        if item in input_value.group(1).strip():
            input_list.append(f"${item}$")
    # 去除空白字符并返回结果
    step_dict = {
        'Purpose': purpose.group(1).strip(),
        'Tool': tool.group(1).strip(),
        'Input': input_list
    }
    return step_dict, flag

def extract_call_content_deny(text):
    # 使用正则表达式提取标签之间的内容
    call_pattern = re.compile(r'<(Call|EndCall|NoCall)>(.*?)</\1>', re.DOTALL)
    call_content = call_pattern.search(text)
    
    tag_type = call_content.group(1)
    content = call_content.group(2)

    # 根据标签类型确定标志
    if tag_type == 'Call':
        flag = 'Call'
    elif tag_type == 'EndCall':
        flag = 'EndCall'
    else:
        flag = 'NoCall'

    # 提取通用的Purpose
    purpose_pattern = re.compile(r'<Purpose>(.*?)</Purpose>', re.DOTALL)
    purpose = purpose_pattern.search(content)

    if flag == 'NoCall':
        # 提取NoCall特有的字段
        category_pattern = re.compile(r'<Category>(.*?)</Category>', re.DOTALL)
        anatomy_pattern = re.compile(r'<Anatomy>(.*?)</Anatomy>', re.DOTALL)
        modality_pattern = re.compile(r'<Modality>(.*?)</Modality>', re.DOTALL)
        ability_pattern = re.compile(r'<Ability>(.*?)</Ability>', re.DOTALL)
        
        category = category_pattern.search(content)
        anatomy = anatomy_pattern.search(content)
        modality = modality_pattern.search(content)
        ability = ability_pattern.search(content)

        step_dict = {
            'Purpose': purpose.group(1).strip(),
            'Category': category.group(1).strip(),
            'Anatomy': anatomy.group(1).strip(),
            'Modality': modality.group(1).strip(),
            'Ability': ability.group(1).strip()
        }
    else:
        # 提取Call和EndCall的Tool和Input
        tool_pattern = re.compile(r'<Tool>(.*?)</Tool>', re.DOTALL)
        input_pattern = re.compile(r'<Input>(.*?)</Input>', re.DOTALL)
        
        tool = tool_pattern.search(content)
        input_value = input_pattern.search(content)

        input_list = []
        for item in ['Image', 'Information', 'Anatomy', 'Modality', 'OrganMask', 'AnomalyMask', 
                     'Disease', 'OrganObject', 'OrganDim', 'OrganQuant', 'AnomalyObject', 
                     'AnomalyDim', 'AnomalyQuant', 'IndicatorName', 'IndicatorValue', 
                     'Report', 'Treatment']:
            if item in input_value.group(1).strip():
                input_list.append(f"${item}$")

        step_dict = {
            'Purpose': purpose.group(1).strip(),
            'Tool': tool.group(1).strip(),
            'Input': input_list
        }

    return step_dict, flag

def find_key_in_nested_dict(target_dict, search_key):
    # 检查当前字典的key
    if search_key in target_dict:
        return True, target_dict[search_key]
    
    # 递归检查所有值为字典的子字典
    for value in target_dict.values():
        if isinstance(value, dict):
            found, result = find_key_in_nested_dict(value, search_key)
            if found:
                return True, result
    
    return False, None

def log_to_file(message, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(str(message) + '\n')

def send_request_gpt(messages, client):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages
    )
    return response.choices[0].message.content

def send_request_gpt4o(messages, client):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages
    )
    return response.choices[0].message.content

def send_request_o1(messages, client):
    response = client.chat.completions.create(
        model="o1-mini",
        messages=messages
    )
    return response.choices[0].message.content

def send_request_claude(messages, client):
    """Sends a request to the model and returns the response content."""
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages
    )
    return response.choices[0].message.content

def send_request_gemini(messages, proxy_url, headers):
    """Sends a request to the model and returns the response content."""
    payload = json.dumps({
        "model": "gemini-1.5-pro-latest",
        "messages": messages
    })
    response = requests.post(proxy_url, headers=headers, data=payload)
    response_content = response.json()
    return response_content['choices'][0]['message']['content']

def send_request_llama(messages, proxy_url, headers):
    """Sends a request to the model and returns the response content."""
    payload = json.dumps({
        "model": "Meta-Llama-3.1-405B-Instruct",
        "messages": messages
    })
    response = requests.post(proxy_url, headers=headers, data=payload)
    response_content = response.json()
    # print(response_content)
    return response_content['choices'][0]['message']['content']

def prepare_decomposition_prompt(TOOLDES):
    """Prepares the decomposition task prompt."""
    decomposition_prompt = f"""
    # Sequential Tool Application Protocol

    ## Task Description
    You are a Radiological Agent who must analyze medical images by sequentially applying appropriate tools. Your task is to:
    1. Follow the tool chain identified in Part 1's decomposition
    2. Execute one tool at a time according to the planned sequence
    3. Use results to determine and refine subsequent specific tool selections
    4. Follow strict input/output formats

    ## Reference from Part 1
    Use the previously identified:
    - Known information categories
    - Tool chain sequence
    This will guide your high-level workflow, but specific tool selection should be refined based on intermediate results.

    ## Available Tools
    The following specific tools are available in your toolbox:

    {TOOLDES}

    ## Response Format
    You must use the following XML-style template for each step:

    For ongoing analysis steps:
    <Call>
        <Purpose>Brief, clear statement of this step's goal</Purpose>
        <Tool>TOOL[number]</Tool>
        <Input>['$variable1$', '$variable2$', ...]</Input>
    </Call>

    For the final step:
    <EndCall>
        <Purpose>Brief, clear statement of this final step's goal</Purpose>
        <Tool>TOOL[number]</Tool>
        <Input>['$variable1$', '$variable2$', ...]</Input>
    </EndCall>

    ## Input Requirements
    1. Required inputs: Must include all mandatory inputs specified in tool description
    2. Optional inputs: Include if available and beneficial to tool performance
    3. Do not include variables that are not relevant to the tool's function
    4. All variables must exist in the current results dictionary
    5. Use proper $$ notation for all variables

    ## Rules and Requirements
    1. Follow the general tool sequence identified in Part 1
    2. Execute exactly one tool per response
    3. Wait for results before proceeding to next step
    4. Only use variables that exist in the provided dictionary
    5. Variables without specific values will be marked as "PLACEHOLDER"
    6. Base specific tool selection on previously received results and Part 1's analysis
    7. Assume the medical image is already provided as $Image$
    8. Only use tools from the provided toolbox
    9. Response should only contain one Call or EndCall xml part
    10. Use <EndCall> only when all necessary steps are complete
    12. Make the best use of suitable tools and exsiting variables to achieve the highest possible performance based on the tool description Performance

    ## Expected Workflow
    1. Reference Part 1's tool chain sequence
    2. Review available information in the results dictionary
    3. Select appropriate specific tool based on Part 1's plan and available data
    4. Identify required and optional inputs for the selected tool
    5. Format response using provided template
    6. Wait for results before proceeding
    7. Repeat until task is complete

    ## Example Response
    For a tool chain from Part 1: [*Organ Segmentation Tool* -> *Anomaly Detection Tool*]
    
    First step response:
    <Call>
        <Purpose>Segment organs in Head and Neck X-ray for anatomical analysis</Purpose>
        <Tool>TOOL3</Tool>
        <Input>['$Image$', '$Anatomy$']</Input>
    </Call>

    Do you understand these instructions? Please indicate your understanding, and then wait for the first task prompt.
    """
    
    decomposition_answer = """
    Yes, I understand the instructions completely. I will:
    - Follow the tool chain sequence from Part 1
    - Execute one tool at a time
    - Include all required inputs for each tool
    - Add optional inputs only when beneficial
    - Use only relevant existing variables
    - Wait for results after each step
    - Base specific tool decisions on Part 1's analysis and available results
    - Respond with only one Call or EndCall XML part
    - Use the suitable tool with the highest performance described when available with appropriate existing input variables to achieve the best performance

    Please provide your first task prompt along with the Part 1 analysis results.
    """
    
    return decomposition_prompt, decomposition_answer

def prepare_decomposition_prompt_deny(TOOLDES):
    """Prepares the decomposition task prompt."""
    decomposition_prompt = f"""
    # Sequential Tool Application Protocol

    ## Task Description
    You are a Radiological Agent who must analyze medical images by sequentially applying appropriate tools. Your task is to:
    1. Follow the tool chain identified in Part 1's decomposition 
    2. Execute one tool at a time according to the planned sequence
    3. Use results to determine and refine subsequent specific tool selections
    4. Check in every step if the tool category is missing when this category of tools is required
    5. Check in every step if the tool is suitable for the detected Anatomy and Modality in reserved value dictionary based on the Tool description Ability and Property
    6. Check in every step if the result in reserved dictinary can be derived from each tool used in former steps based on the limited label list described in Tool Ability.
    7. Follow strict input/output formats

    ## Reference from Part 1
    Use the previously identified:
    - Known information categories
    - Tool chain sequence
    This will guide your high-level workflow, but specific tool selection should be refined based on intermediate results.

    ## Available Tools
    The following specific tools are available in your toolbox. Please carefully check their Ability, Property, Input and Output in the tool description:

    {TOOLDES}

    ## Response Format
    For regular tool execution, use the following XML-style template:

    For ongoing analysis steps:
    <Call>
        <Purpose>Brief, clear statement of this step's goal</Purpose>
        <Tool>TOOL[number]</Tool>
        <Input>['$variable1$', '$variable2$', ...]</Input>
    </Call>

    For the final step:
    <EndCall>
        <Purpose>Brief, clear statement of this final step's goal</Purpose>
        <Tool>TOOL[number]</Tool>
        <Input>['$variable1$', '$variable2$', ...]</Input>
    </EndCall>

    If no suitable tools can complete the required step, use NoCall in three scenarios:

    1. When a category of tools is completely missing:
    <NoCall>
        <Purpose>The purpose requiring a missing tool category</Purpose>
        <Category>The missing category from ['Anatomy Classifier', 'Modality Classifier', 'Organ Segmentor', 'Anomaly Detector', 'Disease Diagnoser', 'Disease Inferencer', 'Biomarker Quantifier', 'Indicator Evaluator', 'Report Generator', 'Treatment Recommender']</Category>
        <Anatomy>Universal</Anatomy>
        <Modality>Universal</Modality>
        <Ability>CategoryMissing</Ability>
    </NoCall>

    2. When specific tools for the required modality-anatomy combination are missing:
    <NoCall>
        <Purpose>The purpose requiring a specific modality-anatomy tool</Purpose>
        <Category>The required tool category</Category>
        <Anatomy>The specific anatomy from ['Universal', 'Head and Neck', 'Chest', 'Breast', 'Abdomen and Pelvis', 'Limb', 'Spine']</Anatomy>
        <Modality>The specific modality from ['Universal', 'X-ray', 'CT', 'MRI', 'Ultrasound']</Modality>
        <Ability>SpecificToolMissing</Ability>
    </NoCall>

    3. When existing tools lack required capabilities:
    <NoCall>
        <Purpose>The purpose requiring advanced capabilities</Purpose>
        <Category>The category of existing but insufficient tools</Category>
        <Anatomy>The relevant anatomy</Anatomy>
        <Modality>The relevant modality</Modality>
        <Ability>InsufficientCapability</Ability>
    </NoCall>

    ## Input Requirements
    1. Required inputs: Must include all mandatory inputs specified in tool description
    2. Optional inputs: Include if available and beneficial to tool performance
    3. Do not include variables that are not relevant to the tool's function
    4. All variables must exist in the current results dictionary
    5. Use proper $$ notation for all variables

    ## Rules and Requirements
    1. Follow the general tool sequence identified in Part 1
    2. Execute exactly one tool per response
    3. Wait for results before proceeding to next step
    4. Only use variables that exist in the provided dictionary
    5. Variables without specific values will be marked as "PLACEHOLDER"
    6. Base specific tool selection on previously received results and Part 1's analysis
    7. Assume the medical image is already provided as $Image$
    8. Only use tools from the provided toolbox
    9. Response should only contain one Call, EndCall, or NoCall xml part
    10. Use <EndCall> only when all necessary steps are complete
    11. If the required category of tools is missing, use the NoCall format to indicate the missing category.
    12. If the Tool description Ability and Property clarify its working Anatomy and Modality, please check if it is suitable for the detected Anatomy and Modality in the reseved dictionary. If no tool is satisfied, use the NoCall format.
    13. If the result in reserved dictionary can not be derived from each tool used in former steps based on the limited label list described in Tool Ability, please use the NoCall format.
    14. Use appropriate <NoCall> format when tools are unavailable, specifying the correct denial reason

    ## Expected Workflow
    1. Reference Part 1's tool chain sequence
    2. Review available information in the results dictionary
    3. Check tool Ability and Property for required step
    4. If tools are insufficient, use appropriate NoCall format
    5. Select appropriate specific tool based on Part 1's plan and available data
    6. Format response using provided template (Call, EndCall, or NoCall)
    7. Wait for results before proceeding
    8. Repeat until task is complete

    ## Example Response
    Regular case:
    <Call>
        <Purpose>Segment organs in Head and Neck X-ray for anatomical analysis</Purpose>
        <Tool>TOOL3</Tool>
        <Input>['$Image$', '$Anatomy$']</Input>
    </Call>

    Missing category case:
    <NoCall>
        <Purpose>Generate structured medical report for findings</Purpose>
        <Category>Report Generator</Category>
        <Anatomy>Universal</Anatomy>
        <Modality>Universal</Modality>
        <Ability>CategoryMissing</Ability>
    </NoCall>

    Missing specific tool case:
    <NoCall>
        <Purpose>Detect early-stage lung nodules in chest X-rays</Purpose>
        <Category>Anomaly Detector</Category>
        <Anatomy>Chest</Anatomy>
        <Modality>X-ray</Modality>
        <Ability>SpecificToolMissing</Ability>
    </NoCall>

    Insufficient capability case:
    <NoCall>
        <Purpose>Detect sub-millimeter brain lesions in MRI</Purpose>
        <Category>Anomaly Detector</Category>
        <Anatomy>Head and Neck</Anatomy>
        <Modality>MRI</Modality>
        <Ability>InsufficientCapability</Ability>
    </NoCall>

    Do you understand these instructions? Please indicate your understanding, and then wait for the first task prompt.
    """
    
    decomposition_answer = """
    Yes, I understand the instructions completely. I will:
    - Follow the tool chain sequence from Part 1
    - Execute one tool at a time
    - Include all required inputs for each tool
    - Add optional inputs only when beneficial
    - Use only relevant existing variables
    - Wait for results after each step
    - Base specific tool decisions on Part 1's analysis and available results
    - If the required category of tools is missing, I will use the NoCall format to indicate the missing category.
    - If Tool Ability and Tool Property clarify its working Anatomy and Modality, I will check if it is suitable for the detected Anatomy and Modality in the reserved dictionary. If not, I will use the NoCall format.
    - If the result in reserved dictionary can not be derived from each tool used in former steps based on the limited label list described in Tool Ability, I will use the NoCall format.
    - Use appropriate <NoCall> format for three distinct cases:
      1. When a tool category is completely missing
      2. When specific tools for a modality-anatomy combination are missing
      3. When existing tools lack required capabilities
    - Respond with only one Call, EndCall, or NoCall XML part

    Please provide your first task prompt along with the Part 1 analysis results.
    """
    
    return decomposition_prompt, decomposition_answer


# def prepare_decomposition_prompt(TOOLDES):
#     """Prepares the decomposition task prompt."""
#     decomposition_prompt = f"""
#     Now your task is to act as a Radiological Agent by decomposing the question's task into a series of 
#     thought chains. You will answer the question step by step by calling different tools. However, the number of tools you can use, their capabilities, and their inputs/outputs are limited. 
#     I will provide you with a toolbox containing the following tools with descriptions:

#     {TOOLDES}

#     Your response should include only one step at a time. Each step must include:

#     The purpose of the step (e.g., determining modality, identifying body part, detecting abnormalities, etc.)
#     The tool from the toolbox that is called for this step
#     The input for this step. Please note that inputs must be values that already exist in the value dictionary.

#     Please use the following template to standardize your output:

#     <Call>
#         <Purpose> Brief, clear statement of the purpose of this tool call. </Purpose>
#         <Tool> Name of the tool called. (named with the number as described in tool description) </Tool>
#         <Input> List all the existing appropriate input variable name to this tool. Using $$ as the special mark. (e.g. ['$Image$', '$Modality$', '$Anatomy$', '$AnomalyObject$']) </Input>
#     </Call>

#     or

#     <EndCall>
#         <Purpose> Brief, clear statement of the purpose of this tool call. </Purpose>
#         <Tool> Name of the tool called. (named with the number as described in tool description) </Tool>
#         <Input> List all the existing appropriate input variable name to this tool. Using $$ as the special mark. (e.g. ['$Image$', '$Modality$', '$Anatomy$', '$AnomalyObject$']) </Input>
#     </EndCall>

#     Important notes:

#     After each tool call, I will provide a dictionary containing known results. You should use these known results to determine which tool to use next. Each key-value pair in the dictionary represents a variable and its value. If a key-value pair has "PLACEHOLDER" as its value, it means that the specific value of this variable is not relevant for subsequent decision-making.

#     Before each tool call, review the dictionary of known results and make the best tool choice based on this information.

#     - The output of each tool call will be provided by me and can be used as input for subsequent tool calls.
#     - Assume that I have already provided you with the radiological image.
#     - After each step, wait for my response before proceeding to the next step.
#     - Your response each time can only use one Call, and each Call can only invoke one tool. If you believe all necessary steps have been completed to answer the question, instead of providing another step, respond with: <EndCall> ... </EndCall> to replace <Call> ... </Call> in the template.
#     - Avoid using additional tools that are not mentioned in the question and are not relevant to the task flow.
    
#     Do you understand?
#     """
#     decomposition_answer = """
#     Yes, I will assume the image is already provided and structure my answer like this:
#     <Call>
#         <Purpose> Segment the organs in the provided X-ray of the head and neck to further analyze the anatomical structures. </Purpose>
#         <Tool> TOOL3 </Tool>
#         <Input> ['$Image$', '$Anatomy$', '$Modality$'] </Input>
#     </Call>
#     Now please provide your first step prompt.
#     """
#     return decomposition_prompt, decomposition_answer

def initialize_reserved_dict(case, initial_input):
    """Initializes the reserved dictionary with initial input information."""
    value_dict = {'$Image$': 'PLACEHOLDER_IMAGE', '$Information$': 'PLACEHOLDER_INFORMATION'}
    score_dict = {'$Image$': 1.0, '$Information$': 1.0}
    try:
        for info in initial_input:
            if info != '$Image$' and info != '$Information$':
                if info == '$IndicatorName$' or info == '$IndicatorValue$':
                    flag, value = find_key_in_nested_dict(case, info.replace('$', '').replace('Indicator', ''))  # IndicatorName -> Name, IndicatorValue -> Value
                else:
                    flag, value = find_key_in_nested_dict(case, info.replace('$', ''))
                assert flag
                value_dict[info] = value
                score_dict[info] = 1.0
    except Exception:
        raise OSError(f"Input {initial_input} is not in Meta Information!")
    fixed_dict = frozendict(score_dict)
    return value_dict, score_dict, fixed_dict

def validate_and_execute_tool(call_dict, value_dict, score_dict, fixed_dict, TOOLBOX):
    """Validates the tool call and executes the corresponding tool function."""
    if call_dict['Tool'] not in TOOLBOX:
        raise NameError(f"Tool {call_dict['Tool']} is not in the toolbox!")
    # print(call_dict['Input'])
    input_list = call_dict['Input']
    # if '$Information$' in input_list:
    #     if '$Information$' not in value_dict:
    #         raise OSError("$Information$ is not in the reserved dictionary!")
    # if '$Anatomy$' in input_list:
    #     if '$Anatomy$' not in value_dict:
    #         raise OSError("$Anatomy$ is not in the reserved dictionary!")
    # if '$Modality$' in input_list:
    #     if '$Modality$' not in value_dict:
    #         raise OSError("$Modality$ is not in the reserved dictionary!")
    # if '$OrganMask$' in input_list:
    #     if '$OrganMask$' not in value_dict:
    #         raise OSError("$OrganMask$ is not in the reserved dictionary!")
    # if '$AnomalyMask$' in input_list:
    #     if '$AnomalyMask$' not in value_dict:
    #         raise OSError("$AnomalyMask$ is not in the reserved dictionary!")
    # if '$Disease$' in input_list:
    #     if '$Disease$' not in value_dict:
    #         raise OSError("$Disease$ is not in the reserved dictionary!")
    # if '$OrganObject$' in input_list:
    #     if '$OrganObject$' not in value_dict:
    #         raise OSError("$OrganObject$ is not in the reserved dictionary!")
    # if '$OrganDim$' in input_list:
    #     if '$OrganDim$' not in value_dict:
    #         raise OSError("$OrganDim$ is not in the reserved dictionary!")
    # if '$OrganQuant$' in input_list:
    #     if '$OrganQuant$' not in value_dict:
    #         raise OSError("$OrganQuant$ is not in the reserved dictionary!")
    # if '$AnomalyObject$' in input_list:
    #     if '$AnomalyObject$' not in value_dict:
    #         raise OSError("$AnomalyObject$ is not in the reserved dictionary!")
    # if '$AnomalyDim$' in input_list:
    #     if '$AnomalyDim$' not in value_dict:
    #         raise OSError("$AnomalyDim$ is not in the reserved dictionary!")
    # if '$AnomalyQuant$' in input_list:
    #     if '$AnomalyQuant$' not in value_dict:
    #         raise OSError("$AnomalyQuant$ is not in the reserved dictionary!")
    # if '$IndicatorName$' in input_list:
    #     if '$IndicatorName$' not in value_dict:
    #         raise OSError("$IndicatorName$ is not in the reserved dictionary!")
    # if '$IndicatorValue$' in input_list:
    #     if '$IndicatorValue$' not in value_dict:
    #         raise OSError("$IndicatorValue$ is not in the reserved dictionary!")
    # if '$Report$' in input_list:
    #     if '$Report$' not in value_dict:
    #         raise OSError("$Report$ is not in the reserved dictionary!")
    # if '$Treatment$' in input_list:
    #     if '$Treatment$' not in value_dict:
    #         raise OSError("$Treatment$ is not in the reserved dictionary!")
    
    # Execute the tool
    value_dict, score_dict = TOOLBOX[call_dict['Tool']](input_list, value_dict, score_dict, fixed_dict)
    tool_called = call_dict['Tool']
    return tool_called, value_dict, score_dict

def update_reserved_dict(case, tool_called_info, value_dict, backend_value_dict, score_dict):
    """Updates the value dictionary with new information."""

    new_keys = set(value_dict.keys()) - set(backend_value_dict.keys())
    flags = {'$Organ$': True, '$Anomaly$': True, '$Disease$': True, 'Biomarker': True, '$Indicator$': True}
    for key in new_keys:
        key_name = key.replace('$','')
        value = None

        if key == '$Report$':
            finding = case['case']['Report']["Finding"]
            impression = case['case']['Report']["Impression"]
            value = f"Findings: {finding} Impression: {impression}"
        elif key in ['$OrganMask$', '$AnomalyMask$']:
            pass  # No action needed
        elif key == '$OrganObject$':
            value = case['case']['OrganBiomarker']['OrganObject']
            if (tool_called_info['Organs'] != None and tool_called_info['Organs'] != []) and value not in tool_called_info['Organs']:
                flags['$Organ$'] = False
        elif key == '$AnomalyObject$':
            value = case['case']['AnomalyBiomarker']['AnomalyObject']
            if (tool_called_info['Anomalies'] != None and tool_called_info['Anomalies'] != []) and value not in tool_called_info['Anomalies']:
                flags['$Anomaly$'] = False
        elif key == ['$Disease$']:
            value = case['case']['Disease']
            if (tool_called_info['Diseases'] != None and tool_called_info['Diseases'] != []) and value not in tool_called_info['Diseases']:
                flags['$Disease$'] = False
        elif key in ['$OrganDim$', '$OrganQuant$']:
            value = case['case']['OrganBiomarker'][key_name]
            if (tool_called_info['Biomarkers'] != None and tool_called_info['Biomarkers'] != []) and value not in tool_called_info['Biomarkers']:
                flags['$Biomarker$'] = False
        elif key in ['$AnomalyDim$', '$AnomalyQuant$']:
            value = case['case']['AnomalyBiomarker'][key_name]
            if (tool_called_info['Biomarkers'] != None and tool_called_info['Biomarkers'] != []) and value not in tool_called_info['Biomarkers']:
                flags['$Biomarker$'] = False
        elif key in ['$IndicatorName$', '$IndicatorValue$']:
            value = case['case']['Indicator'][key_name.replace('Indicator','')]
            if tool_called_info['Indicators'] != None and tool_called_info['Indicators'] != [] and value not in tool_called_info['Indicators']:
                flags['$Indicator$'] = False
        else:
            value = case['case'].get(key_name, None)

        if value is not None:
            value_dict[key] = value

    # Handle special cases for organ and anomaly masks
    if '$OrganMask$' in value_dict and '$OrganObject$' in value_dict:
        value_dict['$OrganObject$'] = case['case']['OrganBiomarker']['OrganObject']
        value_dict['$OrganDim$'] = case['case']['OrganBiomarker']['OrganDim']
        score_dict['$OrganObject$'] = score_dict['$OrganObject$']
        score_dict['$OrganDim$'] = score_dict['$OrganObject$']

    if '$AnomalyMask$' in value_dict and '$AnomalyObject$' in value_dict:
        value_dict['$AnomalyObject$'] = case['case']['AnomalyBiomarker']['AnomalyObject']
        value_dict['$AnomalyDim$'] = case['case']['AnomalyBiomarker']['AnomalyDim']
        score_dict['$AnomalyObject$'] = score_dict['$AnomalyObject$']
        score_dict['$AnomalyDim$'] = score_dict['$AnomalyObject$']
    
    return flags

def prepare_step_prompt(value_dict):
    """Prepares the prompt for the next step based on current results."""
    step_prompt = f"""
    # Next Step Planning

    ## Current Status
    Current results dictionary: {value_dict}

    ## Planning Guidelines
    1. Reference your high-level tool chain from Part 1 decomposition
    2. Consider current results to refine specific tool selection
    3. Maintain sequential progression according to planned workflow
    4. Adjust tool selection if needed based on intermediate results
    5. Make the best use of suitable tools and exsiting variables to achieve the highest possible performance

    ## Input Requirements
    1. Required inputs: Must include all mandatory inputs specified in tool description
    2. Optional inputs: Include if available and beneficial to tool performance
    3. Do not include variables that are not relevant to the tool's function
    4. All variables must exist in the current results dictionary
    5. Use proper $$ notation for all variables

    ## Response Format
    For ongoing analysis (if not final step):
    <Call>
        <Purpose>Brief, clear statement of this step's goal in context of overall analysis</Purpose>
        <Tool>TOOL[number] - must match available specific tools</Tool>
        <Input>['$variable1$', '$variable2$', ...] - use only existing variables from results</Input>
    </Call>

    For final step only:
    <EndCall>
        <Purpose>Brief, clear statement of this final step's goal</Purpose>
        <Tool>TOOL[number] - must match available specific tools</Tool>
        <Input>['$variable1$', '$variable2$', ...] - use only existing variables from results</Input>
    </EndCall>

    ## Format Requirements
    1. Maintain proper XML structure
    2. Use exact tool numbers as specified in tool descriptions
    3. Mark all variables with $$ notation
    4. Include only existing variables from results dictionary
    5. Keep purpose statements clear and concise
    6. Brief response only includes one Call or EndCall XML part without additional explanations

    ## Decision Making Process
    1. Review planned tool chain from Part 1
    2. Check current results in value dictionary
    3. Determine if next planned tool is still appropriate
    4. Refine tool selection if needed based on available data
    5. Format response using appropriate template
    6. Use <EndCall> only if this will be the final step
    7. Choose the tool with the highest performance in the condition of multiple tools suitable with appropriate existing input variables to achieve the best performance

    Please provide your next step based on:
    - Original tool chain plan
    - Current results
    - Available specific tools
    - Remaining analysis needs
    """
    return step_prompt

def prepare_step_prompt_deny(value_dict):
    """Prepares the prompt for the next step based on current results."""
    step_prompt = f"""
    # Next Step Planning

    ## Current Status
    Current results dictionary: {value_dict}

    ## Planning Guidelines
    1. Reference your high-level tool chain from Part 1 decomposition
    2. Consider current results to refine specific tool selection
    3. Maintain sequential progression according to planned workflow
    4. Adjust tool selection if needed based on intermediate results
    5. Check if the tool category is missing when this category of tools is required
    6. Check if the tool is suitable for the detected Anatomy and Modality in reserved value dictionary based on the Tool description Ability and Property
    7. Check if the result in reserved value dictionary can be derived from each tool used in former steps based on the limited label list described in Tool Ability 
    8. If no suitable tool exists, identify which type of denial applies:
       - Missing tool category
       - Missing specific modality-anatomy tool
       - Insufficient tool capability

    ## Input Requirements
    1. Required inputs: Must include all mandatory inputs specified in tool description
    2. Optional inputs: Include if available and beneficial to tool performance
    3. Do not include variables that are not relevant to the tool's function
    4. All variables must exist in the current results dictionary
    5. Use proper $$ notation for all variables

    ## Response Format
    For ongoing analysis (if not final step):
    <Call>
        <Purpose>Brief, clear statement of this step's goal in context of overall analysis</Purpose>
        <Tool>TOOL[number] - must match available specific tools</Tool>
        <Input>['$variable1$', '$variable2$', ...] - use only existing variables from results</Input>
    </Call>

    For final step only:
    <EndCall>
        <Purpose>Brief, clear statement of this final step's goal</Purpose>
        <Tool>TOOL[number] - must match available specific tools</Tool>
        <Input>['$variable1$', '$variable2$', ...] - use only existing variables from results</Input>
    </EndCall>

    When a tool category is completely missing:
    <NoCall>
        <Purpose>The purpose requiring a missing tool category</Purpose>
        <Category>The missing category from ['Anatomy Classifier', 'Modality Classifier', 'Organ Segmentor', 'Anomaly Detector', 'Disease Diagnoser', 'Disease Inferencer', 'Biomarker Quantifier', 'Indicator Evaluator', 'Report Generator', 'Treatment Recommender']</Category>
        <Anatomy>Universal</Anatomy>
        <Modality>Universal</Modality>
        <Ability>CategoryMissing</Ability>
    </NoCall>

    When specific tools for a modality-anatomy combination are missing:
    <NoCall>
        <Purpose>The purpose requiring a specific modality-anatomy tool</Purpose>
        <Category>The required tool category</Category>
        <Anatomy>The specific anatomy from ['Universal', 'Head and Neck', 'Chest', 'Breast', 'Abdomen and Pelvis', 'Limb', 'Spine']</Anatomy>
        <Modality>The specific modality from ['Universal', 'X-ray', 'CT', 'MRI', 'Ultrasound']</Modality>
        <Ability>SpecificToolMissing</Ability>
    </NoCall>

    When existing tools lack required capabilities:
    <NoCall>
        <Purpose>The purpose requiring advanced capabilities</Purpose>
        <Category>The category of existing but insufficient tools</Category>
        <Anatomy>The relevant anatomy</Anatomy>
        <Modality>The relevant modality</Modality>
        <Ability>InsufficientCapability</Ability>
    </NoCall>

    ## Format Requirements
    1. Maintain proper XML structure
    2. Use exact tool numbers as specified in tool descriptions
    3. Mark all variables with $$ notation
    4. Include only existing variables from results dictionary
    5. Keep purpose statements clear and concise
    6. Brief response only includes one Call, EndCall, or NoCall XML part without additional explanations
    7. For NoCall responses, use the appropriate format based on denial type

    ## Decision Making Process
    1. Review planned tool chain from Part 1
    2. Check current results in value dictionary
    3. Check if tool category is missing when this category of tools is required
    4. Check tool Ability and Property in detail to judge its suitability for detected Anatomy and Modality in the value dictionary
    5. Check if the result in reserved value dictionary can be derived from each tool used in former steps based on the limited label list described in Tool Ability
    6. Evaluate tool availability and capability:
       - Is the required tool category available?
       - Are specific tools available for the needed modality-anatomy combination?
       - Do available tools have sufficient capabilities?
    7. If tools are insufficient, use appropriate NoCall format
    8. If tools are available, select and format appropriate Call/EndCall
    9. Use <EndCall> only if this will be the final step

    Please provide your next step based on:
    - Original tool chain plan
    - Current results
    - Available specific tools
    - Remaining analysis needs
    - Tool availability and capability assessment
    """
    return step_prompt

# def prepare_step_prompt(value_dict):
#     """Prepares the prompt for the next step based on current results."""
#     step_prompt = f"""
#     The current results are stored in a dictionary {value_dict}. Please refer to these existing results to plan the next step. Also use the template:

#     <Call>
#         <Purpose> Brief, clear statement of the purpose of this tool call. </Purpose>
#         <Tool> Name of the tool called. (named with the number as described in tool description) </Tool>
#         <Input> List all the existing appropriate input variable name to this tool. Using $$ as the special mark. (e.g. ['$Image$', '$Modality$', '$Anatomy$', '$AnomalyObject$']) </Input>
#     </Call>

#     If you determine that the next step will be the final one, do not use <Call> </Call>. Instead, use <EndCall> </EndCall> as below:

#     <EndCall>
#         <Purpose> Brief, clear statement of the purpose of this tool call. </Purpose>
#         <Tool> Name of the tool called. (named with the number as described in tool description) </Tool>
#         <Input> List all the existing appropriate input variable name to this tool. Using $$ as the special mark. (e.g. ['$Image$', '$Modality$', '$Anatomy$', '$AnomalyObject$']) </Input>
#     </EndCall>

#     When planning your next step:
#     1. If it's not the final step, use the <Call> </Call> format as before.
#     2. If it is the final step, use the <EndCall> </EndCall> format to provide your concluding analysis.

#     Please proceed with planning the next call based on the current results.
#     """
#     return step_prompt

def prepare_conclusion_prompt(value_dict):
    """Prepares the prompt for the final conclusion."""
    conclusion_prompt = f"""
    Based on your Part 1 analysis plan, Part 2 tool execution sequence, and the final results dictionary {value_dict}, provide:

    1. A concise answer to the initial question
    2. Key supporting evidence from your results
    3. How your findings align with the planned analysis

    Keep your response brief and focused on directly answer the initial question.
    """
    return conclusion_prompt

def prepare_valid_prompt(value_dict):
    """Prepares the prompt for validating tool capabilities against results."""
    valid_prompt = f"""
    Current results dictionary: {value_dict}

    Validate if all previously used tools have the necessary capabilities to generate these results.
    
    Compare each tool's documented capabilities with the results in the dictionary. 
    Especially check if the results can be derived from each tool used in former steps based on the limited label list described in Tool Ability. 
    If Yes, respond with <VALID> template, else <INVALID> template.

    Respond strictly using one of these templates:
    <VALID>Yes, all the tools used are reasonable</VALID>
    OR
    <INVALID>No, [specific tool from: 'Organ Segmentor', 'Anomaly Detector', 'Disease Diagnoser', 'Biomarker Quantifier', 'Indicator Evaluator'] does not have the capability to obtain the above results</INVALID>
    """
    return valid_prompt

def saveINFO(info_dict, tool_manager, i, q, logout, messages, base_path):
    """Saves the logs and messages to the specified directories."""
    class_dir = os.path.join(base_path, "class", str(i))
    info_dir = os.path.join(base_path, "info", str(i))
    logs_dir = os.path.join(base_path, "logouts", str(i))
    messages_dir = os.path.join(base_path, "messages", str(i))
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(messages_dir, exist_ok=True)

    with open(os.path.join(class_dir, f"{q}.dill"), "wb") as f:
        dill.dump(tool_manager, f)

    # Save info file
    with open(os.path.join(info_dir, f"{q}.json"), "w") as f:
        json.dump(info_dict, f, indent=4)

    # Save log file
    log_file_path = os.path.join(logs_dir, f"{q}.txt")
    log_to_file(logout, log_file_path)

    # Save messages file
    message_file_path = os.path.join(messages_dir, f"{q}.json")
    with open(message_file_path, "w") as f:
        json.dump(messages, f, indent=4)


def handle_exception(e, info_dict, tool_manager, i, j, q, logout, messages, base_path):
    """Handles exceptions by logging the error and saving current state."""
    logout += f"\nAn error occurred:"
    logout += f"\nError Type: {type(e).__name__}"
    logout += f"\nError Message: {str(e)}"
    logout += "\nTraceback:"

    traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    logout += traceback_str

    saveINFO(info_dict, tool_manager, i, q, logout, messages, base_path)