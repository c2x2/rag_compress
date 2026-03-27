from deepeval.metrics import (
    ContextualPrecisionMetric,
    ArgumentCorrectnessMetric,
    ContextualRecallMetric,
    GEval,
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

def generate_schema(obj):
    if isinstance(obj, dict):
        return {
            "type": "object",
            "properties": {
                k: generate_schema(v) for k, v in obj.items()
            },
            "required": list(obj.keys())
        }
    elif isinstance(obj, list):
        if len(obj) == 0:
            return {"type": "array", "items": {}}
        return {
            "type": "array",
            "items": generate_schema(obj[0])
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

def extract_json_from_prompt(prompt: str):
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

class LocalVLLM(DeepEvalBaseLLM):
    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(model=model_path)  # ⚠️必须调用

        # tokenizer（用于chat template）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    # ✅ 真正加载模型的地方
    def load_model(self):
        return LLM(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.6,
            swap_space=0.5
        )

    # ✅ 核心生成逻辑
    def generate(self, prompt: str) -> str:

        schema = generate_schema(extract_json_from_prompt(prompt))
        guided_decoding = GuidedDecodingParams(
            json=schema
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,   # 评测必须稳定
            max_tokens=2048,
            guided_decoding=guided_decoding
        )
        # 🔥 强制转 chat 格式（避免模型跑偏）
        messages = [
            {"role": "system", "content": "You are a helpful evaluator."},
            {"role": "user", "content": prompt}
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

def main():
    model_path = str(os.getenv("LLM_PATH"))    
    local_model = LocalVLLM(model_path=model_path)
    # contextual_precision = ContextualPrecisionMetric(model=local_model)
    contextual_recall = ContextualRecallMetric(model=local_model, strict_mode=True)
    faithfulness = FaithfulnessMetric(model=local_model, strict_mode=True)
    answerrelevancy = AnswerRelevancyMetric(model=local_model, strict_mode=True)
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the answer is factually correct based on the expected answer.",
        evaluation_params=["input", "actual_output", "expected_output"],
        model=local_model
    )

    test_cases = []

    with open("/home/melonmelon/agent/server/rag_compress/backend/results/langchain/no_compress_triviaqa_web_20260326_002134.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    d = data['data']
    test_case = LLMTestCase(
        input=d["question"][0],
        actual_output=d["answer"][0],
        expected_output=d["ground_truth"][0],
        retrieval_context=d["contexts"][0],
    )
    test_cases.append(test_case)

    result = evaluate(
        test_cases=test_cases,
        metrics=[faithfulness, contextual_recall, answerrelevancy],
        # metrics=[correctness_metric],
        
    )

    for res in result.test_results:
        print(res.input)
        if res.metrics_data:
            for metric in res.metrics_data:
                print(metric.score)
                print(metric.success)
                print(metric.reason)
                print(metric.name)


# def main():
#     text = """
#             {
#                 "verdicts": [
#                     {
#                         "reason": "...",
#                         "verdict": "yes"
#                     },
#                     ...
#                 ]  
#             }
# """
#     res = extract_json_from_prompt(text)
#     print(res)
if __name__ == "__main__":
    main()