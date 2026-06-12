import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import string
from datasets import load_dataset  # 新增：导入加载Hugging Face数据集的库

nltk.download('punkt', quiet=True)

class BM25Retriever:
    def __init__(self, dataset_name: str = "izhx/COMP5423-25Fall-HQ-small"):
        """
        初始化BM25检索器（直接加载老师提供的HQ-small数据集的collection）
        :param dataset_name: Hugging Face数据集名称（固定为老师提供的路径）
        """
        # 1. 加载HQ-small数据集（重点：只取Collection部分作为检索文档库）
        self.dataset = load_dataset(dataset_name, split="collection")
        # 2. 预处理文档（转换为{doc_id, content}格式）
        self.documents = self._preprocess_collection()
        # 3. 构建BM25索引
        self.tokenized_docs = self._preprocess_docs()
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"成功加载 {len(self.documents)} 条Collection文档（来自HQ-small数据集）")

    def _preprocess_collection(self) -> List[Dict]:
        """预处理Collection数据集，提取doc_id和content"""
        docs = []
        for item in self.dataset:
            # 严格匹配老师文档的Collection格式：id和text字段
            if "id" in item and "text" in item:
                docs.append({
                    "doc_id": item["id"],
                    "content": item["text"]
                })
        return docs

    # 以下方法（_preprocess_text、_preprocess_docs、retrieve）完全不变
    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        return [token for token in tokens if len(token) >= 2]

    def _preprocess_docs(self) -> List[List[str]]:
        return [self._preprocess_text(doc["content"]) for doc in self.documents]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        tokenized_query = self._preprocess_text(query)
        if not tokenized_query:
            return [{"doc_id": f"empty_doc_{i+1}", "score": 0.0, "content": "No relevant information"} for i in range(top_k)]
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_doc_indices:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "score": round(doc_scores[idx], 4),
                "content": doc["content"]
            })
        
        while len(results) < top_k:
            fill_doc_id = f"empty_doc_{len(results)+1}"
            results.append({
                "doc_id": fill_doc_id,
                "score": 0.0,
                "content": "No relevant information"
            })
        return results

# 单例实例：全局初始化（直接加载HQ-small的Collection）
retriever = BM25Retriever()