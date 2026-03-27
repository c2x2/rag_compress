# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# def main():
#     model_path = "/home/melonmelon/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"

#     # tokenizer（关键！）
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True
#     )

#     # vLLM
#     llm = LLM(
#         model=model_path,
#         trust_remote_code=True,
#         max_model_len=2048,  # 建议别太大
#         gpu_memory_utilization=0.8,
#         swap_space=0.5
#     )

#     sampling_params = SamplingParams(
#         temperature=0.7,
#         top_p=0.9,
#         max_tokens=100
#     )

#     queries = [
#         "Hello, who are you?",
#         "Where is the capital of France?"
#     ]

#     # 🔥 构造 chat prompt
#     prompts = []
#     for q in queries:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": q}
#         ]

#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         prompts.append(prompt)

#     outputs = llm.generate(prompts, sampling_params)

#     for output in outputs:
#         print("🧠", output.outputs[0].text)
#         pt = len(output.prompt_token_ids)
#         ct = len(output.outputs[0].token_ids)
#         print(pt)
#         print(ct)

# if __name__ == "__main__":
#     main()

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_path="/home/melonmelon/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))