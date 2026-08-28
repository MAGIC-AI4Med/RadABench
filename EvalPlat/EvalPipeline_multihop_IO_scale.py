import argparse
import os
import json
import time
import inspect
from tqdm import tqdm
from Utils.tooluse import ToolManager_IO
from Utils.llm_client import LLMClient, available_models
from Prompting import compose, parse_strategy_list
from requests.exceptions import ProxyError, ConnectionError

from Utils.utils import initial_prompt, select_elements, extract_call_content_deny, find_key_in_nested_dict, log_to_file, prepare_decomposition_prompt_deny, prepare_step_prompt_deny, prepare_conclusion_prompt, prepare_valid_prompt, initialize_reserved_dict, update_reserved_dict, validate_and_execute_tool, saveINFO, handle_exception, prepare_decomposition_prompt, prepare_step_prompt, extract_call_content

class My_Chatbot:
    """Agent core wrapper: one LLM, one simulated tool set, one output tree.

    The model is chosen by alias from ``Configs/models.yaml``; credentials come
    from the environment. See ``.env.example`` for the variables each provider
    needs.
    """

    RUN_TAG = "multiIO_scale"

    def __init__(self, model = "claude-sonnet-4.5", model_path = None, output_root = "Outputs",
                 strategies = ""):
        self.llm = LLMClient(model, model_path = model_path)
        self.chat_func = self.llm.chat
        # Prompting strategies are opt-in; "" reproduces the paper's
        # baseline condition, "all" its integrated condition.
        self.strategy = compose(strategies)
        self.system_prompt = self.strategy.system_prompt(initial_prompt)
        run_tag = self.RUN_TAG + (f"+{'+'.join(parse_strategy_list(strategies))}" if strategies else "")
        self.base_path = os.path.join(output_root, model, run_tag)
        os.makedirs(self.base_path, exist_ok = True)

        entity_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Utils", "entity_dict.json")
        with open(entity_path, "r") as f:
            self.entity_dict = json.load(f)

        # Init Tool
        self.tool_manager = ToolManager_IO()
        self.tool_manager.build_tools_toy()
        self.TOOLINFO = self.tool_manager.toolinfo
        self.TOOLDES = self.tool_manager.tooldes
        self.TOOLBOX = self.tool_manager.toolbox

    def main_work(self, case_json_dir = "Data/combined_casesV3_qa_Subset.json", start = 0, limit = None):
        with open(case_json_dir, "r") as f:
            combined_cases = json.load(f)

        index_list = list(combined_cases.keys())
        stop = len(index_list) if limit is None else min(start + limit, len(index_list))
        for i in range(start, stop):
            qa_dict = combined_cases[index_list[i]]['QA']
            for j in tqdm(range(len(qa_dict)), desc = f"Case {i}"):
                self.pipelinerun(i, j, combined_cases, self.base_path)

    def pipelinerun(self, i, j, combined_cases, base_path):
        index_list = list(combined_cases.keys())
        case_list = list(combined_cases.values())
        tool_manager = ToolManager_IO(case = case_list[i]['case'])
        tool_manager.build_tools_scale()
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
                {"role": "system", "content": self.system_prompt},
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


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Redundant tool set, high level (169 tools)",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default = "claude-sonnet-4.5",
                        help = "model alias from Configs/models.yaml: " + ", ".join(available_models()))
    parser.add_argument("--model-path", default = None,
                        help = "local checkpoint path, for local_hf backends only")
    parser.add_argument("--data", default = "Data/combined_casesV3_qa_Subset.json",
                        help = "patient-record + QA file to evaluate")
    parser.add_argument("--out", default = "Outputs",
                        help = "root directory for logs and per-case traces")
    parser.add_argument("--start", type = int, default = 0,
                        help = "first case index, for resuming an interrupted run")
    parser.add_argument("--limit", type = int, default = None,
                        help = "evaluate at most this many cases (smoke tests)")
    parser.add_argument("--strategies", default = "",
                        help = "prompting strategies, comma separated: "
                               "backprop, reflection, fewshot, multiagent; "
                               "or 'all'. Empty reproduces the baseline.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chatbot = My_Chatbot(model = args.model, model_path = args.model_path,
                         output_root = args.out, strategies = args.strategies)
    chatbot.main_work(case_json_dir = args.data, start = args.start, limit = args.limit)
