from baseline.rag_base import BaseRag
from dotenv import load_dotenv
import os
import json
import langchain
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.callbacks import get_openai_callback
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from typing import Sequence, List, TypedDict
from data_progress.base import QAItem
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
load_dotenv()
os.environ["RAGAS_DO_NOT_TRACK"] = "true" # 禁用匿名追踪，减少网络开销
import logging
# 配置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG)
# 针对 langchain 和 ragas 开启详细日志
logging.getLogger("ragas").setLevel(logging.DEBUG)

# 执行评估
# evaluate(...)

class Ragas(TypedDict):
    question:List[str]
    answer:List[str]
    contexts:List[List[str]]
    ground_truth:List[str]

class LangchainRag(BaseRag):
    llm:HuggingFacePipeline
    embedding:HuggingFaceEmbeddings
    prompt_template:str
    prompt:PromptTemplate
    use_compress:bool
    def __init__(self, name:str, use_compress:bool=False):
        super().__init__(name)
        self.batch_size = 8
        self.top_k = 5
        self.max_context_len = 1200
        self.max_new_tokens = 256
        self.root_path = str(os.getenv("ROOT_PATH"))

        #定义模型
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")
        self.model = model
        model.config.use_cache = False
        #评测用模型
        self.evalute_llm = HuggingFacePipeline(
            pipeline=pipeline(
                task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=self.max_new_tokens, 
                batch_size=self.batch_size, 
                max_length=None, 
                return_full_text=False,
                temperature=0,
                do_sample=False
            )
        )
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

        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
            return_tensors="pt"
        ).to("cuda")

        # 记录输入张量的宽度（这是所有 prompt 补齐后的统一长度）
        input_tensor_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                # temperature=0,
                # top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        responses = []
        prompt_tokens_counts = []
        completion_tokens_counts = []
        total_tokens_counts = []

        for i in range(len(outputs)):
            # 1. 获取该行真正的 prompt 长度（不含 padding）
            actual_prompt_len = inputs["attention_mask"][i].sum().item()
            
            # 2. 正确切片：从 input_tensor_len 之后开始才是真正的生成内容
            # 无论前面有多少 PAD，生成的回答始终跟在整个输入张量后面
            generated_tokens = outputs[i][input_tensor_len:]
            
            # 3. 过滤掉生成结果末尾可能的填充 token (如果有的话)
            # 寻找第一个 EOS token 之后的位置并截断，或者直接用 skip_special_tokens
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 4. 计算真实的 Token 数量
            # 生成的有效 token 数需要排除掉生成阶段产生的 [PAD]
            # 我们通过解码后再编码，或者直接统计非 pad 的 token
            actual_gen_tokens = (generated_tokens != self.tokenizer.pad_token_id).sum().item()

            responses.append(text)
            prompt_tokens_counts.append(actual_prompt_len)
            completion_tokens_counts.append(actual_gen_tokens)
            total_tokens_counts.append(actual_prompt_len + actual_gen_tokens)

        return responses, prompt_tokens_counts, completion_tokens_counts, total_tokens_counts

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
        filepath = f"{self.root_path}/results/{self.name}/{self.__get_compress()}_{self.dataname}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"结果已保存至{filepath}")
        return
