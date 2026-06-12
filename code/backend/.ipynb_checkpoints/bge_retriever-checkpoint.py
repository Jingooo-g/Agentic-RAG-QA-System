import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import time # Added for BGEVectorizer progress

# Ensure nltk data is downloaded if needed by other parts, though not directly used in BGE.
# nltk.download('punkt', quiet=True) # Not needed for BGEVectorizer or VectorSearcher

# 单例实例：全局初始化
# 注意：这里的模型路径和向量数据库路径需要根据你的实际环境进行配置。
# 示例路径（请根据实际情况修改）：
# BGE模型通常较大，建议下载到本地并指定本地路径。如果使用Hugging Face模型ID，transformers库会自动下载。
# 向量数据库文件需要提前生成，格式为JSONL，每行一个JSON对象，包含"id"和"embedding"字段。
# 例如：{"id": "doc1", "embedding": [0.1, 0.2, ...]}
_default_bge_model_path = "BAAI/bge-base-en-v1.5" # 或者本地路径如 "c:/path/to/your/bge-model"
from datasets import load_dataset

# Load model directly
tokenizer = AutoTokenizer.from_pretrained(_default_bge_model_path)
model = AutoModel.from_pretrained(_default_bge_model_path)

class BGEVectorizer:
    def __init__(self, device: str = None):
        """
        初始化BGE向量化器

        Args:
            device: 运行设备，默认为自动选择 (cuda 或 cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        global tokenizer, model # Use global tokenizer and model
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        print(f"BGE模型加载完成，使用设备: {self.device}")
        self.instruction = "为这个句子生成表示："

    def get_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        将文本列表转换为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            numpy数组，每行是一个文本的向量
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(texts),
                desc="生成文本向量",
                unit="text",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

        start_time = time.time()

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_start_time = time.time()

                batch_with_instruction = [self.instruction + text for text in batch_texts]

                encoded_input = self.tokenizer(
                    batch_with_instruction,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                batch_time = time.time() - batch_start_time

                if show_progress and pbar:
                    pbar.update(len(batch_texts))
                    elapsed_time = time.time() - start_time
                    processed_texts = min(i + batch_size, len(texts))
                    if processed_texts > 0:
                        texts_per_second = processed_texts / elapsed_time
                        remaining_texts = len(texts) - processed_texts
                        remaining_time = remaining_texts / texts_per_second if texts_per_second > 0 else 0

                        pbar.set_postfix({
                            "批次": f"{i // batch_size + 1}/{total_batches}",
                            "速度": f"{texts_per_second:.1f}文本/秒",
                            "剩余时间": f"{remaining_time:.1f}秒",
                            "设备": self.device
                        })

        finally:
            if show_progress and pbar:
                pbar.close()

        total_time = time.time() - start_time
        if show_progress:
            print(f"向量生成完成！总共处理 {len(texts)} 个文本，耗时 {total_time:.2f} 秒")
            print(f"平均速度: {len(texts) / total_time:.2f} 文本/秒")

        return np.vstack(all_embeddings)

    def encode_text(self, sentence: str) -> np.ndarray:
        """
        简化版的单句编码函数

        Args:
            sentence: 要编码的句子
        Returns:
            句子向量
        """
        # Use self.device instead of re-detecting
        device = self.device

        text_with_instruction = self.instruction + sentence

        encoded_input = self.tokenizer(
            text_with_instruction,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            token_embeddings = model_output[0]
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()[0]

    def mean_pooling(self, model_output, attention_mask):
        """
        使用平均池化从模型输出中获取句子嵌入
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class VectorSearcher:
    def __init__(self, vector_db: Dict[str, List[float]]):
        """
        初始化向量搜索器

        Args:
            vector_db: 内存中的向量数据库 (字典，键为id，值为向量)
        """
        self.vector_db = vector_db
        self.vector_ids = list(self.vector_db.keys())
        self.vectors = np.array([self.vector_db[vid] for vid in self.vector_ids])
        print(f"向量数据库加载完成，共 {len(self.vector_db)} 个向量")

    def search_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的前K个

        Returns:
            最相似的向量id和分数的列表
        """
        if not self.vectors.size: # Handle empty vector database
            return [{"doc_id": f"empty_doc_{i+1}", "score": 0.0, "content": "No relevant information"} for i in range(top_k)]

        similarities = cosine_similarity([query_vector], self.vectors)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id = self.vector_ids[idx]
            score = round(similarities[idx], 4)
            # In a real scenario, you'd fetch the content from a document store using doc_id
            # For now, we'll just return doc_id and score, similar to BM25Retriever's output structure
            results.append({
                "doc_id": doc_id,
                "score": score,
                "content": "Content not available directly from vector DB" # Placeholder
            })
        
        # Fill up to top_k if fewer results are found (e.g., if vector_db has < top_k entries)
        while len(results) < top_k:
            fill_doc_id = f"empty_doc_{len(results)+1}"
            results.append({
                "doc_id": fill_doc_id,
                "score": 0.0,
                "content": "No relevant information"
            })
        return results

class BGERetriever:
    def __init__(self, dataset_name: str = "izhx/COMP5423-25Fall-HQ-small", device: str = None):
        """
        初始化BGE检索器

        Args:
            dataset_name: Hugging Face数据集名称
            device: 运行设备，默认为自动选择 (cuda 或 cpu)
        """
        self.vectorizer = BGEVectorizer(device=device)
        self.documents = self._load_and_preprocess_dataset(dataset_name)
        
        # 生成所有文档的嵌入向量
        doc_contents = [doc["content"] for doc in self.documents]
        print(f"开始为 {len(doc_contents)} 篇文档生成嵌入向量...")
        embeddings = self.vectorizer.get_embeddings(doc_contents, show_progress=True)
        
        # 构建内存中的向量数据库
        vector_db = {}
        for i, doc in enumerate(self.documents):
            vector_db[doc["doc_id"]] = embeddings[i].tolist()
            
        self.searcher = VectorSearcher(vector_db=vector_db)
        print(f"BGE检索器初始化完成，数据集: {dataset_name}")

    def _load_and_preprocess_dataset(self, dataset_name: str) -> List[Dict]:
        """
        加载并预处理Hugging Face数据集
        """
        print(f"正在加载Hugging Face数据集: {dataset_name}...")
        dataset = load_dataset(dataset_name, split="collection")
        docs = []
        for item in dataset:
            if "id" in item and "text" in item:
                docs.append({"doc_id": item["id"], "content": item["text"]})
        print(f"数据集加载完成，共 {len(docs)} 篇文档。")
        return docs

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        根据查询检索最相关的文档

        Args:
            query: 查询字符串
            top_k: 返回最相关的文档数量

        Returns:
            包含文档ID、分数和内容的字典列表
        """
        query_vector = self.vectorizer.encode_text(query)
        retrieved_docs = self.searcher.search_similar_vectors(query_vector, top_k=top_k)
        
        # 将content从self.documents中补充到retrieved_docs
        doc_id_to_content = {doc["doc_id"]: doc["content"] for doc in self.documents}
        for doc in retrieved_docs:
            if doc["doc_id"] in doc_id_to_content:
                doc["content"] = doc_id_to_content[doc["doc_id"]]
            else:
                doc["content"] = "Content not found in original dataset."
        
        return retrieved_docs

# 单例实例：全局初始化
# 注意：这里的模型路径和向量数据库路径需要根据你的实际环境进行配置。
# 示例路径（请根据实际情况修改）：
# BGE模型通常较大，建议下载到本地并指定本地路径。如果使用Hugging Face模型ID，transformers库会自动下载。
# 向量数据库文件需要提前生成，格式为JSONL，每行一个JSON对象，包含"id"和"embedding"字段。
# 例如：{"id": "doc1", "embedding": [0.1, 0.2, ...]}

try:
    retriever = BGERetriever()
except Exception as e:
    print(f"初始化BGE检索器失败: {e}")
    print("请确保BGE模型已下载到指定路径，并且向量数据库文件存在且格式正确。")
    retriever = None # Set to None if initialization fails