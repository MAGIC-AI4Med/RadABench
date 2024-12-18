import json
import time
import torch
import inspect
from tqdm import tqdm
from openai import OpenAI
from Utils.tooluse import ToolManager_IO
from requests.exceptions import ProxyError, ConnectionError
from transformers import AutoModelForCausalLM, AutoTokenizer

from Utils.utils import initial_prompt, select_elements, extract_call_content_deny, find_key_in_nested_dict, log_to_file, prepare_decomposition_prompt_deny, prepare_step_prompt_deny, prepare_conclusion_prompt, prepare_valid_prompt, initialize_reserved_dict, update_reserved_dict, validate_and_execute_tool, saveINFO, handle_exception, send_request_gpt, send_request_claude, send_request_gemini, send_request_o1, send_request_gpt4o, send_request_llama, prepare_sigle_hop_prompt

class My_Chatbot:
    def __init__(self, chatbot_type = "GPT-4", model_path_or_api = "sk-Be86e11ZdWdO0IHOGr4cLVutkROL6xpwRBPoEEViVLU20SzA"):
        #Init Part
        if chatbot_type == "GPT-4":
            self.model_or_client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key="sk-EE7JpikARsXP66nZzQLTH5OT2WiqLEajuPIByPtZhO2tKNej"
            )
            self.tokenizer = None
            self.chat_func = self.GPT_chat_function
            self.base_path = "GPT4/General"

        if chatbot_type == "GPT-4o":
            self.model_or_client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key="sk-EE7JpikARsXP66nZzQLTH5OT2WiqLEajuPIByPtZhO2tKNej"
            )
            self.tokenizer = None
            self.chat_func = self.GPT4o_chat_function
            self.base_path = "GPT4o/General"

        if chatbot_type == "o1-mini":
            self.model_or_client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key="sk-EE7JpikARsXP66nZzQLTH5OT2WiqLEajuPIByPtZhO2tKNej"
            )
            self.tokenizer = None
            self.chat_func = self.o1_chat_function
            self.base_path = "o1/General"
        
        if chatbot_type == "Claude-3.5":
            self.model_or_client = OpenAI( 
                base_url="https://chat.cloudapi.vip/v1/",
                api_key="sk-7B4k9bjiYPIkawKQaPWNw7eqnt8oWtJUoKKJutYJWsKK8atJ",
                default_headers={"x-foo": "true"}
            )
            self.tokenizer = None
            self.chat_func = self.Claude_chat_function
            self.base_path = "Claude/General"

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
            self.base_path = "Gemini/General"

        if chatbot_type == "LLAMA":
            Baseurl = "https://api.claudeshop.top"
            Skey = "sk-oiOwo8jOL1nLRI2NLnfBcWVz7jQ1Y39hJFLYH3hZgvbDTeis"

            self.proxy_url = Baseurl + "/v1/chat/completions"
            self.headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {Skey}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            self.tokenizer = None
            self.chat_func = self.llama_chat_function
            self.base_path = "LLAMA/General"

        if chatbot_type == "Mixtral":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_api)
            self.model_or_client = AutoModelForCausalLM.from_pretrained(model_path_or_api, device_map="auto", torch_dtype=torch.float16,attn_implementation="sdpa")
            self.chat_func = self.Mistral_chat_function
            self.base_path = "Mixtral/General"
        
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
            self.base_path = "Qwen2.5/General"
            
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
        for i in range(20,30):
            index_list = list(combined_cases.keys())
            qa_dict =  combined_cases[index_list[i]]['QA']
            for j in tqdm(range(0,len(qa_dict)), desc="Processing"):
                self.pipelinerun(i, j, combined_cases, self.base_path)
    
    def pipelinerun(self, i, j, combined_cases, base_path):
        index_list = list(combined_cases.keys())
        case_list = list(combined_cases.values())
        TOOLINFO = self.TOOLINFO
        TOOLDES = self.TOOLDES
        TOOLBOX = self.TOOLBOX
        TOOLBOX_INFO = {tool_name: inspect.getsource(tool) for tool_name, tool in TOOLBOX.items()}
        tool_manager = self.tool_manager
        TOOL_dict = {"TOOLINFO": TOOLINFO, "TOOLDES": TOOLDES, "TOOLBOX": TOOLBOX_INFO}
        try:
            index, case = index_list[i], case_list[i]
            query_index_list = list(case['QA'].keys())
            QA_type = query_index_list[j]
            query = case['QA'][QA_type]['Q']

            logout = f"Case {i}, Question {j}, {QA_type}: {query}\n"

            prompt = prepare_sigle_hop_prompt(TOOLDES)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
            decomposition = self.chat_func(messages)
            logout += f"Decomposition:\n {decomposition}\n"
            saveINFO(TOOL_dict, tool_manager, i, QA_type, logout, messages, base_path)
        except (ProxyError, ConnectionError, KeyError) as e:
            print(f"An error occurred: {e}")
            time.sleep(2)
            handle_exception(e, TOOL_dict, tool_manager, i, j, QA_type, logout, messages, base_path)

if __name__=="__main__":
    chatbot = My_Chatbot(chatbot_type = "LLAMA", model_path_or_api = None)
    chatbot.main_work()