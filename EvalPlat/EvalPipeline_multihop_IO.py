import json
import time
import torch
import inspect
from tqdm import tqdm
from openai import OpenAI
from Utils.tooluse import ToolManager_IO
from requests.exceptions import ProxyError, ConnectionError
from transformers import AutoModelForCausalLM, AutoTokenizer

from Utils.utils import initial_prompt, select_elements, extract_call_content_deny, find_key_in_nested_dict, log_to_file, prepare_decomposition_prompt_deny, prepare_step_prompt_deny, prepare_conclusion_prompt, prepare_valid_prompt, initialize_reserved_dict, update_reserved_dict, validate_and_execute_tool, saveINFO, handle_exception, send_request_gpt, send_request_claude, send_request_gemini, send_request_o1, send_request_gpt4o, send_request_llama, prepare_decomposition_prompt, prepare_step_prompt, extract_call_content

class My_Chatbot:
    def __init__(self, chatbot_type = "GPT-4", model_path_or_api = "sk-Be86e11ZdWdO0IHOGr4cLVutkROL6xpwRBPoEEViVLU20SzA"):
        #Init Part
        if chatbot_type == "GPT-4":
            self.model_or_client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key="sk-l7FBRt6eH9Kh9sx7pjGTWr6rLiC9C46P7ORY7mykvAubCRmX"
            )
            self.tokenizer = None
            self.chat_func = self.GPT_chat_function
            self.base_path = "GPT4/multiIO"

        if chatbot_type == "GPT-4o":
            self.model_or_client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key="sk-IyJeBtiaRHiG8PgJZI1xlaX0aKiq0pai937uvKzTZEfLEhAb"
            )
            self.tokenizer = None
            self.chat_func = self.GPT4o_chat_function
            self.base_path = "GPT4o//multiIO"

        if chatbot_type == "o1-mini":
            self.model_or_client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key="sk-7rJ2cgxeqYnq2AcVKuUQFb8ydvxwFLZE2fcCredPBwATZXgv"
            )
            self.tokenizer = None
            self.chat_func = self.o1_chat_function
            self.base_path = "o1/multiIO"
        
        if chatbot_type == "Claude-3.5":
            self.model_or_client = OpenAI( 
                base_url="https://chat.cloudapi.vip/v1/",
                api_key="sk-gnLrtiUgnkTqIm68w56VK78C8HDHWUUwkT31XTRUDfAZE008",
                default_headers={"x-foo": "true"}
            )
            self.tokenizer = None
            self.chat_func = self.Claude_chat_function
            self.base_path = "Claude/multiIO"

        if chatbot_type == "Gemini-1.5":
            Baseurl = "https://api.claudeshop.top"
            Skey = "sk-YKIsf5FHQhY3NdQW9f6944Dd2d5742049d7cDb398e66B862"

            self.proxy_url = Baseurl + "/v1/chat/completions"
            self.headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {Skey}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            self.tokenizer = None
            self.chat_func = self.Gemini_chat_function
            self.base_path = "Gemini/multiIO"

        if chatbot_type == "LLAMA":
            Baseurl = "https://api.claudeshop.top"
            Skey = "sk-VFvBuKu3ap1UROhoMwgfTmd5Z3VpOYBqCtjXE6r2X7n9mTpw"

            self.proxy_url = Baseurl + "/v1/chat/completions"
            self.headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {Skey}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            self.tokenizer = None
            self.chat_func = self.llama_chat_function
            self.base_path = "LLAMA/multiIO"

        if chatbot_type == "Mixtral":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_api)
            self.model_or_client = AutoModelForCausalLM.from_pretrained(model_path_or_api, device_map="auto", torch_dtype=torch.float16,attn_implementation="sdpa")
            self.chat_func = self.Mistral_chat_function
            self.base_path = "Mixtral/multiIO"
        
        if chatbot_type == "Qwen": 
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_or_client = AutoModelForCausalLM.from_pretrained(
                model_path_or_api,
                torch_dtype=torch.float16,
                device_map = 'auto',
                attn_implementation="sdpa"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_api)
            self.chat_func = self.Qwen_chat_function
            self.base_path = "Qwen2.5/multiIO"
            
        with open("Utils/entity_dict.json", "r") as f:
            self.entity_dict = json.load(f)
        
        # Init Tool
        self.tool_manager = ToolManager_IO()
        self.tool_manager.build_tools_toy()
        self.TOOLINFO = self.tool_manager.toolinfo
        self.TOOLDES = self.tool_manager.tooldes
        self.TOOLBOX = self.tool_manager.toolbox

    def GPT_chat_function(self, messages):
        
        output = send_request_gpt(messages, self.model_or_client)  
        return output

    def GPT4o_chat_function(self, messages):
        
        output = send_request_gpt4o(messages, self.model_or_client)  
        return output
    
    def o1_chat_function(self, messages):
        
        output = send_request_o1(messages, self.model_or_client)  
        return output

    def Claude_chat_function(self, messages):

        output = send_request_claude(messages, self.model_or_client)
        return output
    
    def Gemini_chat_function(self, messages):

        output = send_request_gemini(messages, self.proxy_url, self.headers)
        return output

    def llama_chat_function(self, messages):

        output = send_request_llama(messages, self.proxy_url, self.headers)
        return output

    def Mistral_chat_function(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model_or_client.device)
        outputs = self.model_or_client.generate(inputs, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    def Qwen_chat_function(self, messages):
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model_or_client.device)

        generated_ids = self.model_or_client.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def main_work(self, case_json_dir = "CASE/combined_casesV3_qa_Subset.json"):
        with open(case_json_dir, "r") as f:
            combined_cases = json.load(f)

        # for i in range(0,len(combined_cases)):
        for i in range(0,20):
            index_list = list(combined_cases.keys())
            qa_dict =  combined_cases[index_list[i]]['QA']
            for j in tqdm(range(len(qa_dict)), desc="Processing"):
                self.pipelinerun(i, j, combined_cases, self.base_path)
        

    def pipelinerun(self, i, j, combined_cases, base_path):
        index_list = list(combined_cases.keys())
        case_list = list(combined_cases.values())
        tool_manager = ToolManager_IO(case = case_list[i]['case'])
        tool_manager.build_tools_toy()
        TOOLINFO = tool_manager.toolinfo
        TOOLDES = tool_manager.tooldes
        TOOLBOX = tool_manager.toolbox
        TOOLBOX_INFO = {tool_name: inspect.getsource(tool) for tool_name, tool in TOOLBOX.items()}
        TOOL_dict = {"TOOLINFO": TOOLINFO, "TOOLDES": TOOLDES, "TOOLBOX": TOOLBOX_INFO}

        try:
            index, case = index_list[i], case_list[i]
            query_index_list = list(case['QA'].keys())
            QA_type = query_index_list[j]
            query = case['QA'][QA_type]['Q']

            logout = f"Case {i}, Question {j}, {QA_type}: {query}\n"

            messages = [
                {"role": "system", "content": initial_prompt},
                {"role": "user", "content": query}
            ]

            initial_output = None
            initial_output = self.chat_func(messages)

            logout += f"Initial Output: {initial_output}\n"

            initial_addition_list = []
            # for item in ['$Anatomy$', '$Modality$', '$Disease$', '$OrganObject$', '$OrganDim$', '$OrganQuant$', '$AnomalyObject$', '$AnomalyDim$', '$AnomalyQuant$', '$IndicatorName$', '$IndicatorValue$', '$Report$', '$Treatment$']:
            for item in ['$Anatomy$', '$Modality$', '$Disease$']:
                if item in initial_output:
                    initial_addition_list.append(item)
            initial_input = list(set(['$Image$', '$Information$'] + initial_addition_list))
            # initial_input = ['$Image$', '$Information$']
            value_dict, score_dict, fixed_dict = initialize_reserved_dict(case, initial_input)
            #[*Anatomy Classification Tool*, *Modality Classification Tool*, *Organ Segmentation Tool*, *Anomaly Detection Tool*, *Disease Diagnosis Tool*, *Disease Inference Tool*, *Organ Biomarker Quantification Tool*, *Anomaly Biomarker Quantification Tool*, *Indicator Evaluation Tool*, *Report Generation Tool*, *Treatment Recommendation Tool*]
            tool_list = []
            for item in ['Anatomy Classification Tool', 'Modality Classification Tool', 'Organ Segmentation Tool', 'Anomaly Detection Tool', 'Disease Diagnosis Tool', 'Disease Inference Tool', 'Organ Biomarker Quantification Tool', 'Anomaly Biomarker Quantification Tool', 'Indicator Evaluation Tool', 'Report Generation Tool', 'Treatment Recommendation Tool']:
                if item in initial_output:
                    tool_list.append(item)
            #join each item with '->'
            tool_chain = ' -> '.join(tool_list)

            logout += f"Initial Value Dict: {value_dict}\nInitial Score Dict: {score_dict}\nInitial Fixed Dict: {fixed_dict}\n"
            logout += f"High-level Tool chain: {tool_chain}\n"

            decomposition_prompt, decomposition_answer = prepare_decomposition_prompt(TOOLDES)
            
            messages.extend([
                {"role": "assistant", "content": f"The known information is: {value_dict}\n The high-level tool chain is: {tool_chain}\n"},
                {"role": "user", "content": decomposition_prompt},
                {"role": "assistant", "content": decomposition_answer}
            ])

            logout += f"\nMemory bank: {value_dict} \nScore bank: {score_dict} \nFixed bank: {fixed_dict}"
            count = 0
            Flag = True

            while Flag:

                step_prompt = prepare_step_prompt(value_dict)
                messages.append({"role": "user", "content": step_prompt})
                count += 1

                logout += f"\nStep {count} starts!"

                # Send step request and get the step output
                step_output = self.chat_func(messages)

                logout += f"\nStepoutput: {step_output}"

                backend_value_dict = value_dict.copy()
                # Extract call content and update Flag
                call_dict, Flag = extract_call_content(step_output)

                logout += f"\nCall Dict: {call_dict}"

                # Validate and execute tool call
                tool_called, value_dict, score_dict = validate_and_execute_tool(call_dict, value_dict, score_dict, fixed_dict, TOOLBOX)
                tool_called_info = TOOLINFO[tool_called]
                # Update reserved dictionary
                flags = update_reserved_dict(case, tool_called_info, value_dict, backend_value_dict, score_dict)

                logout += f"\nMemory bank: {value_dict} \nScore bank: {score_dict} \nFixed bank: {fixed_dict}"
                logout += f"\nStep {count} completes!"

                messages.append({"role": "assistant", "content": step_output})
            
            logout += f"Final combination starts!"
            value_dict['$Information$'] = case['case']['Information']
            conclusion_prompt = prepare_conclusion_prompt(value_dict)
            messages.append({"role": "user", "content": conclusion_prompt})
            conclusion_output = self.chat_func(messages)

            logout += f"\nConclusion: {conclusion_output}\n"
            
            saveINFO(TOOL_dict, tool_manager, i, QA_type, logout, messages, base_path)
        
        except (ProxyError, ConnectionError, KeyError) as e:
            print(f"An error occurred: {e}")
            time.sleep(2)
            self.pipelinerun(i, j, combined_cases, self.base_path)
        except Exception as e:
            handle_exception(e, TOOL_dict, tool_manager, i, j, QA_type, logout, messages, base_path)

if __name__=="__main__":
    chatbot = My_Chatbot(chatbot_type = "LLAMA", model_path_or_api = None)
    chatbot.main_work()