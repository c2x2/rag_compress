import jieba
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class DynamicSentenceCompressor:
    def __init__(self, query_boost=5.0):
        """
        初始化压缩器
        :param query_boost: Query命中词的权重放大系数 (即公式中的 lambda)
        """
        self.query_boost = query_boost
        # 初始化 TF-IDF 向量化器，使用 jieba 进行分词
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None)

    def _split_sentences(self, text):
        """简单的中文句子切分"""
        text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
        text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
        text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
        text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
        sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 5] # 过滤掉太短的无意义短句
        return sentences

    def compress(self, query: str, context: str, compression_ratio: float = 0.5) -> str:
        """
        核心压缩逻辑
        :param query: 用户查询
        :param context: RAG 召回的原始合并文本
        :param compression_ratio: 期望保留的比例 (0.0 到 1.0)，0.5表示保留50%的句子
        """
        # 1. 句子切分
        sentences = self._split_sentences(context)
        if not sentences:
            return context

        # 2. 动态拟合局部的 TF-IDF (仅基于当前召回的 context)
        # 这里我们将每一个句子视为一个 "Document"，来计算局部 IDF
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        word2index = {word: i for i, word in enumerate(feature_names)}
        
        # 3. 解析 Query，获取 Query 包含的词集
        query_words = set(jieba.lcut(query))

        # 4. 对每个句子进行打分
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = jieba.lcut(sentence)
            if not words:
                sentence_scores.append(0.0)
                continue
            
            score = 0.0
            valid_word_count = 0
            
            for word in words:
                if word in word2index:
                    # 获取该词的 TF-IDF 值
                    tfidf_val = tfidf_matrix[i, word2index[word]]
                    
                    # 如果词在 Query 中出现，给予权重奖励
                    boost = self.query_boost if word in query_words else 0.0
                    
                    score += tfidf_val * (1.0 + boost)
                    valid_word_count += 1
            
            # 使用词数进行归一化，防止长句霸榜
            normalized_score = score / valid_word_count if valid_word_count > 0 else 0.0
            sentence_scores.append(normalized_score)

        # 5. 根据压缩比例进行筛选
        num_to_keep = max(1, int(len(sentences) * compression_ratio))
        
        # 获取得分最高的 top_k 个句子的索引
        top_indices = np.argsort(sentence_scores)[-num_to_keep:]
        
        # 必须按原文本的顺序重新排序，保证逻辑连贯
        top_indices = sorted(top_indices)

        # 6. 重组文本
        compressed_text = " ... ".join([sentences[idx] for idx in top_indices])
        return compressed_text

# =========================================
# 测试运行代码
# =========================================
if __name__ == "__main__":
    # 模拟用户 Query
    test_query = "多智能体系统中，如何解决长期规划的崩溃问题？"
    
    # 模拟 RAG 召回的冗长文本块 (包含废话和核心信息)
    test_context = """
    近年来，人工智能领域取得了巨大的进展。我们发现，大语言模型在很多自然语言处理任务上表现出色。
    然而，在多智能体系统中，长期规划的崩溃问题一直是一个核心痛点。
    这是因为随着交互轮数的增加，智能体的上下文窗口会被无关信息填满。
    为了解决这个问题，研究人员提出了动态角色演化的机制。
    动态角色演化允许智能体根据任务进度适时地改变自身的设定，从而剥离掉历史的包袱。
    其实今天的天气也很不错，非常适合出去散步。
    通过引入记忆剪枝和状态机回溯，长期规划的稳定性得到了显著提升。
    综上所述，这是一个非常有前景的研究方向。
    """

    compressor = DynamicSentenceCompressor(query_boost=5.0)
    
    # 尝试保留 40% 的核心句子
    result = compressor.compress(query=test_query, context=test_context, compression_ratio=0.4)
    
    print("【原始 Query】:", test_query)
    print("\n【压缩前文本长度】:", len(test_context))
    print("【压缩后文本长度】:", len(result))
    print("\n【压缩后内容】:\n", result)