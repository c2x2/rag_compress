from deepeval.metrics import(
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import json
from deepeval.models.base_model import DeepEvalBaseLLM
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer
load_dotenv()
import re

class LocalVLLM(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(model=model_path)  # ⚠️必须调用

        # tokenizer（用于chat template）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    def generate_schema(self, obj):
        if isinstance(obj, dict):
            return {
                "type": "object",
                "properties": {
                    k: self.generate_schema(v) for k, v in obj.items()
                },
                "required": list(obj.keys())
            }
        elif isinstance(obj, list):
            if len(obj) == 0:
                return {"type": "array", "items": {}}
            return {
                "type": "array",
                "items": self.generate_schema(obj[0])
            }
        elif isinstance(obj, str):
            return {"type": "string"}
        elif isinstance(obj, int):
            return {"type": "integer"}
        elif isinstance(obj, float):
            return {"type": "number"}
        elif isinstance(obj, bool):
            return {"type": "boolean"}
        else:
            return {}

    def extract_json_from_prompt(self, prompt: str):
        stack = []
        start = None

        for i, char in enumerate(prompt):
            if char == '{':
                if not stack:
                    start = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        json_str = prompt[start:i+1]
                        json_str = re.sub(r"\.\.\.", "", json_str)
                        
                        # 去掉多余逗号（比如 ,] 或 ,}）
                        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
                        try:
                            return json.loads(json_str)
                        except:
                            pass
        return None

    # ✅ 真正加载模型的地方
    def load_model(self):
        return LLM(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9
        )

    # ✅ 核心生成逻辑
    def generate(self, prompt: str) -> str:
        # ✅ 1. 提取 schema
        schema = self.generate_schema(self.extract_json_from_prompt(prompt))

        guided_decoding = GuidedDecodingParams(
            json=schema
        )

        # ✅ 2. 控制生成行为（重点）
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096*2,                  # ❗别再 10000 了，会炸
            repetition_penalty=1.2,           # ✅ 防复读
            guided_decoding=guided_decoding
        )

        # ✅ 3. 强约束 prompt（关键！）
        strict_prompt = prompt + """

    IMPORTANT:
    - The number of verdicts MUST equal the number of input statements.
    - Do NOT repeat identical verdicts.
    - Stop immediately after completing the JSON.
    - Output ONLY valid JSON.
    """

        messages = [
            {"role": "system", "content": "You are a strict JSON evaluator."},
            {"role": "user", "content": strict_prompt}
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.model.generate([chat_prompt], self.sampling_params)
        result = outputs[0].outputs[0].text

        return result

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Local-VLLM-Qwen"

class EvalEngine:
    def __init__(self, rag_name) -> None:
        model_path = str(os.getenv("LLM_PATH_2"))    
        self.model = LocalVLLM(model_path=model_path)
        self.metrics = {
            # "answer_relevancy":AnswerRelevancyMetric(model=self.model, strict_mode=True),
            # "faithfulness":FaithfulnessMetric(model=self.model, strict_mode=True),
            # "contextual_recall":ContextualRecallMetric(model=self.model, strict_mode=True)
        }
        self.root_path = str(os.getenv("ROOT_PATH"))
        self.name = rag_name
    
    def load_data(self, file_path):
        self.resultname = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['data']
        
        self.test_cases = []
        query_counts = len(self.data['question'])        
        for i in range(query_counts):
            self.test_cases.append(
                LLMTestCase(
                    input=self.data['question'][i],
                    actual_output=self.data['answer'][i],
                    expected_output=self.data['ground_truth'][i],
                    retrieval_context=[context[:4096] for context in self.data['contexts'][i]]
                )
            )
    
    def eval(self):
        result = evaluate(
            test_cases=self.test_cases,
            metrics=list(self.metrics.values())
        )

        evaluate_results = {}
        for res in result.test_results:
            res_metric={}
            if res.metrics_data:
                for metric in res.metrics_data:
                    res_metric[metric.name] = metric.score
            evaluate_results[res.input] = res_metric
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = f"{self.root_path}/backend/results/{self.name}/evaluate_results/{self.resultname}_{timestamp}_evaluate_result.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluate_results, f)
        
        print(f"结果已经保存至{filepath}")