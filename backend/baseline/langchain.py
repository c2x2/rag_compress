from backend.baseline.rag_base import BaseRag
from dotenv import load_dotenv
import os
import json
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline
from typing import Sequence, List, TypedDict
from vllm import LLM, SamplingParams
from backend.data_progress.base import QAItem
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall
)
from datasets import Dataset
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
load_dotenv()
os.environ["RAGAS_DO_NOT_TRACK"] = "true" # 禁用匿名追踪，减少网络开销


# 执行评估
# evaluate(...)

class Ragas(TypedDict):
    question:List[str]
    answer:List[str]
    contexts:List[List[str]]
    ground_truth:List[str]

class LangchainRag(BaseRag):
    def __init__(self, use_compress:bool=False, tokenizer=None, model=None):
        super().__init__("langchain")
        self.batch_size = 8
        self.top_k = 5
        self.max_context_len = 1200
        self.max_new_tokens = 256
        self.root_path = str(os.getenv("ROOT_PATH"))

        #定义模型
        model_path = str(os.getenv("LLM_PATH"))
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer = tokenizer
        if not model:
            model = LLM(
                model=model_path,
                dtype="float16",
                gpu_memory_utilization=0.8,   # 🔥吃满显存
                max_model_len=2048
            )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens
        )        
        self.model = model
        #评测用模型
        # self.evalute_llm = HuggingFacePipeline(
        #     pipeline=pipeline(
        #         task="text-generation", 
        #         model=model, 
        #         tokenizer=tokenizer, 
        #         max_new_tokens=self.max_new_tokens, 
        #         batch_size=self.batch_size, 
        #         max_length=None, 
        #         return_full_text=False,
        #         temperature=0,
        #         do_sample=False
        #     )
        # )
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(os.getenv("EMBEDDING_PATH")), 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={
                "batch_size": self.batch_size,   # 🔥 核心优化
                "normalize_embeddings": True
            }
        )
        self.use_compress = use_compress

    def __get_compress(self) -> str:
        return "use_compress" if self.use_compress else "no_compress"

    def load_data(self, datasets: List[QAItem], dataname:str):
        self.datasets = datasets
        self.dataname = dataname

    def measure(self, origin=None, compressed=None):
        print("============开始评测结果============")
        filepath = f"{self.root_path}/results/{self.name}/{self.__get_compress()}_{self.dataname}.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)

        MAX_TOKENS=self.max_context_len
        def truncate_fields(example):
            # 截断 context (ragas 的 context 通常是 list)
            if "contexts" in example and example["contexts"]:
                # 简单的字符串截断（粗略估计：1个 token 约等于 3-4 个字符，或者直接按字符数硬截）
                # 稳妥起见，如果单个 context 极长，进行截断
                example["contexts"] = [c[:MAX_TOKENS * 2] for c in example["contexts"]]
            
            # 截断 answer 和 user_input
            if "answer" in example and example["answer"]:
                example["answer"] = example["answer"][:MAX_TOKENS * 2]
            if "user_input" in example and example["user_input"]:
                example["user_input"] = example["user_input"][:MAX_TOKENS * 2]
            return example
        #计算answer质量
        raw_dataset = Dataset.from_dict(results['data'])
        # 使用 map 进行并行截断处理
        dataset = raw_dataset.map(truncate_fields)

        answer_quality = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_correctness,
                context_recall
            ],
            llm=self.evalute_llm,
            embeddings=self.embedding,
            batch_size=2
        ).to_pandas().to_dict()

        target_path = f"{self.root_path}/results/{self.name}/{self.__get_compress()}_{self.dataname}_quality.json"
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(answer_quality, f, indent=4)

        print(f"评测结果保存至{target_path}")

        #计算压缩后和压缩前比较tokens降低程度
        # origin_tokens = np.array(origin["tokens"]["total_tokens"])
        # compressed_tokens = np.array(compressed["tokens"]["total_tokens"])

        # compression_ratio = (origin_tokens - compressed_tokens) / np.maximum(origin_tokens, 1)
        # mean_sample_radio = np.mean(compression_ratio)
        # mean_dataset_radio = np.mean(compression_ratio)
        
        # return{
        #     "compressed_results":compressed_results.to_pandas().to_dict(),
        #     "origin_results":origin_results.to_pandas().to_dict(),
        #     "mean_sample":float(mean_sample_radio),
        #     "mean_dataset":float(mean_dataset_radio)
        # }


    def compress(self, docs:List[Document]) -> List[Document]:
        compressed_docs = []
        for doc in docs:
            compressed_docs.append(Document(doc.page_content[:50]))
        
        return compressed_docs
    
    def build_prompt(self, context, query):
        return f"""
                Given these texts:
                -----
                {context}
                -----
                Please answer the following question:
                {query}
                """

    def generate_batch(self, prompts):
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(text)
        
        outputs = self.model.generate(
            formatted_prompts,
            self.sampling_params
        )

        responses = []
        prompt_tokens = []
        completion_tokens = []
        total_tokens = []

        for out in outputs:
            text = out.outputs[0].text
            responses.append(text)

            # 🔥 vLLM自带token统计
            pt = len(out.prompt_token_ids)
            ct = len(out.outputs[0].token_ids)
            tt = pt + ct

            prompt_tokens.append(pt)
            completion_tokens.append(ct)
            total_tokens.append(tt)

        return responses, prompt_tokens, completion_tokens, total_tokens

    def run(self):
        """
        对每个qa进行问答测试
        """
        print("============开始运行rag系统============")
        target: Ragas = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        token_stats = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }

        for i in tqdm(range(0, len(self.datasets), self.batch_size), desc="start query"):
            batch = self.datasets[i:i+self.batch_size]

            batch_prompts = []
            batch_docs = []
            batch_qas = []

            for qa in batch:
                retriever = InMemoryVectorStore.from_texts(
                    qa["contents"], self.embedding
                ).as_retriever(search_kwargs={"k": self.top_k})

                docs = retriever.invoke(qa["query"])

                if self.use_compress:
                    docs = self.compress(docs)

                # 🔥 控制context长度
                context_text = "\n".join([d.page_content for d in docs])
                context_text = context_text[:self.max_context_len]

                prompt = self.build_prompt(context_text, qa["query"])

                batch_prompts.append(prompt)

                batch_docs.append(docs)
                batch_qas.append(qa)

            responses, ptks, ctks, ttks = self.generate_batch(batch_prompts)

            # 写入
            for qa, docs, resp, pt, ct, tt in zip(
                batch_qas, batch_docs, responses, ptks, ctks, ttks
            ):

                target["question"].append(qa["query"])
                target["contexts"].append([d.page_content for d in docs])
                target["ground_truth"].append(qa["answer"])
                target["answer"].append(resp)

                token_stats["prompt_tokens"].append(int(pt))
                token_stats["completion_tokens"].append(int(ct))
                token_stats["total_tokens"].append(int(tt))

            torch.cuda.empty_cache()

        final_output = {
            "data": target,
            "tokens": token_stats
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.root_path}/backend/results/rag_results/{self.name}/{self.__get_compress()}_{self.dataname}_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"结果已保存至{filepath}")
        return
    
    def llm_model_func(
        self,
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs
    ):

        # 🌿 1. 构造 messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in history_messages:
            messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        # 🌿 2. 构造输入文本（Qwen chat格式）
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 🌿 3. tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]

        # 🌙 4. 推理
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # 🌿 5. 计算生成 tokens
        generated_tokens = outputs[0][prompt_tokens:]
        completion_tokens = generated_tokens.shape[-1]

        # 🌿 6. decode
        result = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        total_tokens = prompt_tokens + completion_tokens

        return result, prompt_tokens, completion_tokens, total_tokens
    
    def run_demo(self, contexts:list[str], query:str, use_compress:bool):
        retriever = InMemoryVectorStore.from_texts(contexts, self.embedding).as_retriever(search_kwargs={"k": self.top_k})

        docs = retriever.invoke(query)
        if use_compress:
            docs = self.compress(docs)

        prompt = self.build_prompt(docs, query)

        responses, ptks, ctks, ttks = self.llm_model_func(prompt)

        return {
            "answer":responses,
            "tokens":{
                "Prompt":ptks,
                "Completion":ctks,
                "Total":ttks
            }
        }