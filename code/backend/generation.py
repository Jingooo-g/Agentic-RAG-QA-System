import json
import sys
from typing import List, Dict, Optional
import re
import math
import os
import requests
from tqdm import tqdm

MY_API_KEY = "sk-zwzhzgrwaqagcagxfyxhobouyycpuqpwkjbsfwbgxcevrueg"
MY_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 1. Retrieval Module Initialization (Dual Retriever)
try:
    # Attempt relative import (for app.py execution)
    from .retrieval import VectorSearcher, BGEVectorizer, BM25Retriever
except ImportError:
    # Fallback import (for direct script execution)
    from retrieval import VectorSearcher, BGEVectorizer, BM25Retriever

print(">>> Initializing Retrievers...")
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Data folder is next to generation.py -> backend/data/
DATA_DIR = os.path.join(current_dir, "data")

# 3. Construct specific file paths
VECTOR_DB_PATH = os.path.join(DATA_DIR, "bge_vector_db.jsonl")
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")

# --- A. Initialize Dense Retriever (BGE) ---
vector_searcher = None
query_encoder = None
doc_text_map = {}

if os.path.exists(VECTOR_DB_PATH):
    print(f"[Dense] Loading Vector DB...")
    vector_searcher = VectorSearcher(VECTOR_DB_PATH)
    query_encoder = BGEVectorizer()

    # Load text map (required for Dense retrieval)
    print("[Dense] Loading document text map (mapping ID to Content)...")
    count = 0
    with open(VECTOR_DB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                doc_id = item.get("id")
                text = item.get("text", item.get("content", ""))
                if doc_id:
                    doc_text_map[doc_id] = text
                    count += 1
            except:
                continue
    print(f"[Dense] Successfully loaded {count} documents into text map.")
else:
    print(f"[Error] Vector DB NOT FOUND at: {VECTOR_DB_PATH}")
    print("Please run 'python backend/retrieval.py' to generate it.")

# --- B. Initialize Sparse Retriever (BM25) ---
if os.path.exists(COLLECTION_PATH):
    print(f"[Sparse] Initializing BM25 Retriever from {COLLECTION_PATH}...")
    bm25_retriever_instance = BM25Retriever(local_file=COLLECTION_PATH)
else:
    print(f"[Error] Collection file NOT FOUND at: {COLLECTION_PATH}")
    bm25_retriever_instance = None


# --- C. Define Unified Dispatcher ---
class RetrieverDispatcher:
    def retrieve(self, query: str, method: str = "dense", top_k: int = 10) -> List[Dict]:
        method = method.lower()

        # Temporary lists
        dense_results = []
        sparse_results = []

        # 1. Execute Dense Retrieval
        if "dense" in method or "hybrid" in method:
            if vector_searcher and query_encoder:
                # print(f"DEBUG: Running Dense Retrieval...")
                # Get embedding (first element of array)
                query_vec = query_encoder.get_embeddings([query], show_progress=False)[0]
                results = vector_searcher.search_similar_vectors(query_vec, top_k=top_k)
                for doc_id, score in results:
                    content = doc_text_map.get(doc_id, "")
                    dense_results.append({
                        "doc_id": doc_id, "id": doc_id,
                        "score": float(score),
                        "content": content, "text": content
                    })

        # 2. Execute Sparse Retrieval
        if "sparse" in method or "hybrid" in method:
            # print(f"DEBUG: Running Sparse Retrieval...")
            sparse_results = bm25_retriever_instance.retrieve(query, top_k=top_k)

        # 3. Return Logic
        # Mode A: Dense Only
        if method == "dense":
            return dense_results

        # Mode B: Sparse Only
        if method == "sparse":
            return sparse_results

        # Mode C: Hybrid (Simple Deduplication Merge)
        if "hybrid" in method:
            # Simple strategy: deduplication merge
            combined = []
            seen_ids = set()

            # Prioritize Dense results
            for doc in dense_results:
                if doc['doc_id'] not in seen_ids:
                    combined.append(doc)
                    seen_ids.add(doc['doc_id'])

            # Supplement with unique Sparse results
            for doc in sparse_results:
                if doc['doc_id'] not in seen_ids:
                    # Note: No score normalization here, simple append for coverage
                    combined.append(doc)
                    seen_ids.add(doc['doc_id'])

            # Truncate to top_k
            return combined[:top_k]

        return dense_results  # Default fallback


# Global retriever instance
retriever = RetrieverDispatcher()


# 2. RAG Core Logic

class BaseGenerator:
    """Base Generator Class (Consistent with generation_module.py interface)"""

    def generate(self, query: str, retrieved_docs: List[Dict], max_new_tokens: int = 256) -> str:
        raise NotImplementedError


class EntityProcessor:
    """Entity Processing Module Wrapper"""

    @staticmethod
    def clean(token: str) -> str:
        token = token.strip()
        token = re.sub(r'[^A-Za-z0-9\s\'-]', '', token)
        return token

    @staticmethod
    def extract_main_entity(text: str, answer: Optional[str] = None) -> str:
        # If the answer is unknown, do not use it for entity extraction to avoid unrelated words being mistakenly extracted from the explanation paragraph.
        if answer and "i don't know" in answer.lower():
            answer = None
        candidates = []
        exclude_words_for_proper_nouns = {"who", "what", "where", "when", "why", "how", "and", "the", "a", "an", "i",
                                          "don", "know", "yes", "no", "of", "in", "with", "for", "on", "at", "by",
                                          "from", "up", "about", "into", "over", "after", "beneath", "below", "between",
                                          "through", "above", "around", "among", "across", "behind", "before", "down",
                                          "off", "out", "under", "upon", "within", "without", "throughout", "is", "was",
                                          "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
                                          "will", "would", "shall", "should", "can", "could", "may", "might", "must",
                                          "are", "this", "that", "these", "those", "he", "she", "it", "they", "we",
                                          "you", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its",
                                          "our", "their", "to", "as", "but", "or", "nor", "yet", "so", "if", "unless",
                                          "until", "while", "when", "where", "why", "how", "than", "then", "once",
                                          "though", "although", "even", "because", "since", "before", "after", "while",
                                          "whether", "else", "such", "only", "just", "also", "too", "very", "really",
                                          "quite", "rather", "almost", "always", "never", "often", "sometimes",
                                          "usually", "seldom", "hardly", "scarcely", "barely", "ever", "once", "twice",
                                          "thrice", "again", "further", "more", "most", "less", "least", "much", "many",
                                          "few", "little", "some", "any", "no", "every", "all", "both", "each",
                                          "either", "neither", "one", "two", "three", "four", "five", "six", "seven",
                                          "eight", "nine", "ten", "first", "second", "third", "fourth", "fifth",
                                          "other", "another", "such", "what", "which", "who", "whom", "whose", "where",
                                          "when", "why", "how", "here", "there", "then", "now"}
        # Exclude common descriptive or transitional words to prevent them from being mistakenly selected as entities from explanatory text.
        extra_excludes = {"additionally", "explanation", "provided", "evidence", "therefore", "insufficient", "based",
                          "solely", "given", "cannot", "explanation:"}
        banned_starters = {"are", "is", "was", "were", "do", "does", "did", "can", "could", "would", "should", "which",
                           "what", "who", "whom", "whose", "where", "when", "why", "how"}

        if answer:
            for m in re.findall(r'"([^"]+)"', answer):
                token = EntityProcessor.clean(m)
                if len(token.split()) >= 1 and len(token) > 2:
                    candidates.append(token)
            proper_ans = re.findall(r'([A-Z][a-zA-Z-]+(?:\s[A-Z][a-zA-Z-]+)*)', answer)
            for m in proper_ans:
                token = EntityProcessor.clean(m)
                if token.lower() not in exclude_words_for_proper_nouns and token.lower() not in extra_excludes and len(
                        token) > 2:
                    candidates.append(token)
        for m in re.findall(r'"([^"]+)"', text):
            token = EntityProcessor.clean(m)
            if len(token.split()) >= 1 and len(token) > 2:
                candidates.append(token)
        proper = re.findall(r'([A-Z][a-zA-Z-]+(?:\s[A-Z][a-zA-Z-]+)*)', text)
        for m in proper:
            # Remove interrogative/auxiliary verb prefixes, e.g., "Are John" -> "John"
            m2 = re.sub(r'^(Are|Is|Do|Does|Did|Can|Could|Would|Should|Which|What|Who|Whom|Whose|Where|When|Why|How)\s+',
                        '', m)
            token = EntityProcessor.clean(m2)
            if token.lower() not in exclude_words_for_proper_nouns and token.lower() not in extra_excludes and len(
                    token) > 2:
                candidates.append(token)

        # Filter candidates starting with interrogative/auxiliary verbs
        candidates = [c for c in candidates if not re.match(r'^(?:' + '|'.join(banned_starters) + r')\b', c.lower())]
        candidates = [c for c in candidates if len(c) > 1]
        if not candidates:
            return ""

        # Prioritize using entities from the answer (if valid), otherwise use entities extracted from the question.
        if answer:
            answer_candidates = [EntityProcessor.clean(m) for m in
                                 re.findall(r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)', answer)
                                 if EntityProcessor.clean(
                    m).lower() not in exclude_words_for_proper_nouns and EntityProcessor.clean(
                    m).lower() not in extra_excludes and len(EntityProcessor.clean(m)) > 2]
            answer_candidates = [c for c in answer_candidates if
                                 not re.match(r'^(?:' + '|'.join(banned_starters) + r')\b', c.lower())]
            if answer_candidates:
                def _pos(s, t):
                    m = re.search(re.escape(s), t, flags=re.IGNORECASE)
                    return m.start() if m else 10 ** 9

                return min(answer_candidates, key=lambda c: _pos(c, answer))

        entities_from_text = EntityProcessor.extract_entities(text)
        if entities_from_text:
            def _pos_text(s, t):
                m = re.search(re.escape(s), t, flags=re.IGNORECASE)
                return m.start() if m else 10 ** 9

            return min(entities_from_text, key=lambda c: _pos_text(c, text))

        # Fallback: Select based on the earliest occurrence position in the question, length is only used for tie-breaking.
        def _pos_any(s, t):
            m = re.search(re.escape(s), t, flags=re.IGNORECASE)
            return m.start() if m else 10 ** 9

        best = min(candidates, key=lambda c: (_pos_any(c, text), -len(c)))
        return best

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extracts all candidate proper noun entities from the text (deduplicated, supports hyphens; filters stop words)."""
        exclude = {"who", "what", "which", "where", "when", "why", "how", "and", "the", "a", "an", "are", "is", "was",
                   "were", "do", "does", "did", "can", "could", "would", "should", "of", "in", "on", "at", "by", "from",
                   "into", "over", "after", "beneath", "below", "between", "through", "above", "around", "among",
                   "across", "behind", "before", "down", "off", "out", "under", "upon", "within", "without",
                   "throughout", "to", "as", "but", "or", "nor", "yet", "so", "if", "unless", "until", "while", "than",
                   "then", "once", "though", "although", "even", "because", "since", "whether", "else", "only", "just",
                   "also", "very", "both", "either", "vs", "versus", "during", "about"}
        extra_excludes = {"additionally", "explanation", "provided", "evidence", "therefore", "insufficient", "based",
                          "solely", "given", "cannot", "explanation:"}
        proper = re.findall(r'([A-Z][a-zA-Z-]+(?:\s[A-Z][a-zA-Z-]+)*)', text)
        cleaned = []
        for m in proper:
            # Remove interrogative/auxiliary verb prefixes (added Was/Were, etc.)
            m2 = re.sub(
                r'^(Are|Is|Was|Were|Do|Does|Did|Can|Could|Would|Should|Which|What|Who|Whom|Whose|Where|When|Why|How)\s+',
                '', m)
            tok = EntityProcessor.clean(m2)
            if tok and tok.lower() not in exclude and tok.lower() not in extra_excludes:
                cleaned.append(tok)
        uniq = []
        for c in cleaned:
            if c not in uniq:
                uniq.append(c)
        # Sort by length, prioritizing longer entities.
        return sorted(uniq, key=len, reverse=True)


class PromptBuilder:
    """Prompt Builder Module Wrapper"""

    @staticmethod
    def build_prompt(question: str, retrieved_docs: List[Dict], model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                     reasoning_steps: Optional[List[str]] = None, forbid_unknown: bool = False) -> str:
        context = "\n".join([f"[Doc{d['doc_id']} score={d['score']}] {d['content']}" for d in retrieved_docs if
                             d.get('content') and d.get('content') != 'No relevant information'])
        question_lower = question.lower()

        specific_instruction = ""

        # 1. Comparison Queries (MUST BE FIRST)
        # Ensure this highest priority instruction is not overwritten by subsequent general instructions
        if "are both what" in question_lower or "share" in question_lower:
            specific_instruction = (
                "You MUST return the single, most concise, common noun or phrase "
                "that describes both entities (e.g., 'magazine', not 'Black Enterprise is a magazine')."
            )

        # 2. General WH- Questions
        # Only execute if specific_instruction is empty, avoiding overwriting comparison instruction
        if not specific_instruction:
            if "where" in question_lower:
                specific_instruction = " Provide a specific location or place name. "
            elif "when" in question_lower:
                specific_instruction = " Provide a specific time, date, or year. "
            elif "who" in question_lower:
                specific_instruction = " Provide a specific person's name. "
            elif "what" in question_lower:
                specific_instruction = " Provide a clear and specific answer. "

        # 3. Yes/No Questions - Last priority
        # Apply Yes/No constraint only for strict Yes/No questions, not overwriting comparison instruction
        is_yes_no = bool(re.search(r'^\s*(?:are|is|do|does|did|can)\b', question_lower)) or (
                " both " in question_lower) or (" either " in question_lower)
        if is_yes_no and not specific_instruction:
            specific_instruction = " Answer exactly 'Yes' or 'No' when applicable. "

        steps_instruction = ""
        if reasoning_steps:
            formatted = "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(reasoning_steps)])
            steps_instruction = (
                "\nFollow these reasoning steps carefully before answering:\n"
                f"{formatted}\n"
                "State the final answer concisely after finishing the reasoning.\n"
            )
        # Unknown policy: Try to answer if there is evidence; only say "I don't know" if completely irrelevant.
        unknown_policy = "Only say 'I don't know' if none of the context mentions the queried entities or attributes."
        if forbid_unknown:
            unknown_policy = "Do NOT answer 'I don't know'; when any context mentions the queried entities or attributes, produce the best-supported concise answer."

        prompt = (
            f"You are a helpful assistant. Answer the question based only on the provided context. Provide a concise answer, preferably a phrase or a single entity, without extra explanations or conversational fillers.{specific_instruction}"
            f"{steps_instruction}"
            f"{unknown_policy}\n"
            f"Context:\n{context}\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt

    @staticmethod
    def build_prompt_with_facts(question: str, retrieved_docs: List[Dict], model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                reasoning_steps: Optional[List[str]] = None, forbid_unknown: bool = False,
                                known_facts: Optional[List[str]] = None) -> str:
        context = "\n".join([f"[Doc{d['doc_id']} score={d['score']}] {d['content']}" for d in retrieved_docs if
                             d.get('content') and d.get('content') != 'No relevant information'])
        facts_section = ""
        if known_facts:
            facts_section = "Known facts:\n" + "\n".join([f"- {f}" for f in known_facts]) + "\n"
        question_lower = question.lower()
        specific_instruction = ""
        is_yes_no = bool(re.search(r'^\s*(?:are|is|do|does|did|can)\b', question_lower)) or (
                    " both " in question_lower) or (" either " in question_lower)
        if is_yes_no:
            specific_instruction = " Answer exactly 'Yes' or 'No' when applicable. "
        elif "where" in question_lower:
            specific_instruction = " Provide a specific location or place name. "
        elif "when" in question_lower:
            specific_instruction = " Provide a specific time, date, or year. "
        elif "who" in question_lower:
            specific_instruction = " Provide a specific person's name. "
        elif "what" in question_lower:
            specific_instruction = " Provide a clear and specific answer. "
        steps_instruction = ""
        if reasoning_steps:
            formatted = "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(reasoning_steps)])
            steps_instruction = (
                "\nFollow these reasoning steps carefully before answering:\n"
                f"{formatted}\n"
                "State the final answer concisely after finishing the reasoning.\n"
            )
        unknown_policy = "Only say 'I don't know' if none of the context mentions the queried entities or attributes."
        if forbid_unknown:
            unknown_policy = "Do NOT answer 'I don't know'; when any context mentions the queried entities or attributes, produce the best-supported concise answer."
        prompt = (
            f"You are a helpful assistant. Answer the question based only on the provided context and known facts. Provide a concise answer, preferably a phrase or a single entity, without extra explanations or conversational fillers.{specific_instruction}"
            f"{steps_instruction}"
            f"{unknown_policy}\n"
            f"Context:\n{facts_section}{context}\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt


class DocProcessor:
    """Document Processing Module Wrapper"""

    @staticmethod
    def rerank_docs(question: str, docs: List[Dict]) -> List[Dict]:
        question_lower = question.lower()
        keywords = []
        if "where" in question_lower: keywords.append("where")
        if "when" in question_lower: keywords.append("when")
        if "who" in question_lower: keywords.append("who")
        # Extra: Boost score for documents containing entities mentioned in the question.
        entities = EntityProcessor.extract_entities(question)
        if not keywords and not entities:
            return docs
        boosted_docs = []
        for doc in docs:
            bonus = 0.0
            content_lower = doc.get("content", "").lower()
            for kw in keywords:
                if kw in content_lower:
                    bonus += 2.5
            for ent in entities:
                if ent and ent.lower() in content_lower:
                    bonus += 3.5
            boosted_docs.append({**doc, "adjusted_score": doc["score"] + bonus})
        boosted_docs.sort(key=lambda d: d.get("adjusted_score", d["score"]), reverse=True)
        return [{k: v for k, v in doc.items() if k != "adjusted_score"} for doc in boosted_docs]

    @staticmethod
    def prune_context_documents(docs: List[Dict], max_doc: int = 10) -> List[Dict]:
        return docs[:max_doc]

    @staticmethod
    def rerank_docs_by_subqueries(question: str, sub_queries: List[str], docs: List[Dict]) -> List[Dict]:
        # Reranks documents based on entities/keywords from both the original and sub-queries.
        q_lower = question.lower()
        keywords = []
        for kw in ["where", "when", "who", "what"]:
            if kw in q_lower:
                keywords.append(kw)
        # Merge entities from the original question and sub-queries
        entities = set(EntityProcessor.extract_entities(question))
        for sq in sub_queries:
            for e in EntityProcessor.extract_entities(sq):
                entities.add(e)
        if not keywords and not entities:
            return docs
        boosted_docs = []
        for doc in docs:
            bonus = 0.0
            content_lower = doc.get("content", "").lower()
            for kw in keywords:
                if kw in content_lower:
                    bonus += 2.5
            for ent in entities:
                if ent and ent.lower() in content_lower:
                    bonus += 4.0
            boosted_docs.append({**doc, "adjusted_score": doc["score"] + bonus})
        boosted_docs.sort(key=lambda d: d.get("adjusted_score", d["score"]), reverse=True)
        return [{k: v for k, v in doc.items() if k != "adjusted_score"} for doc in boosted_docs]


class MultiTurnManager:
    """Multi-Turn Context Management Module Wrapper"""

    def __init__(self):
        self.multi_turn_contexts = {}

    def init_multi_turn_context(self, session_id: str):
        if session_id not in self.multi_turn_contexts:
            self.multi_turn_contexts[session_id] = []

    def rewrite_query(self, session_id: str, cur_query: str) -> str:
        ctx = self.multi_turn_contexts.get(session_id, [])
        if not ctx:
            return cur_query

        rewritten = cur_query
        cur_query_lower = cur_query.lower()
        last_turn = ctx[-1] if ctx else None

        # 1. Get all entities and answer data
        last_answer = last_turn.get("answer", "")
        last_entity = last_turn.get("main_entity", "")
        last_question_entity = last_turn.get("question_entity", "")
        # last_answer_entity is the entity extracted by EntityProcessor.extract_main_entity from the answer
        last_answer_entity = last_turn.get("answer_entity", "")

        # Set human target (He/She) - prioritize entity from the question
        human_target = last_question_entity or last_entity

        # Set object target (It/Its) - prioritize entity from the answer
        non_person_target = last_answer_entity

        # If entity extraction fails (last_answer_entity is empty), and the previous answer is short (like an album/movie title), use the literal answer text as the object.
        if not non_person_target and 2 < len(last_answer.split()) <= 5:
            if last_answer and last_answer[0].isupper():
                non_person_target = EntityProcessor.clean(last_answer)

        # If non_person_target is still empty, and current query asks for 'it',
        # we skip replacing 'it' with human_target, as a person is not an object.

        pronoun_triggers = [" it ", " it?", " it.", " he ", " she ", " they ", " him ", " them "]
        possessive_triggers = [" his ", " her ", " their ", " its "]
        relation_terms = "(wife|First Lady|husband|spouse|mother|father|son|daughter|brother|sister|partner|girlfriend|boyfriend)"

        if any(p in f" {cur_query_lower} " for p in (pronoun_triggers + possessive_triggers)):

            # --- Human Pronoun and Relation Handling (He/She/His) ---
            if human_target:
                # Handle relational phrases: his/her/their + relative
                rewritten = re.sub(rf"\b(his|her|their)\s+{relation_terms}\b",
                                   lambda m: f"{human_target}'s {m.group(2)}", rewritten, flags=re.IGNORECASE)
                if re.search(rf"\b{relation_terms}\b", rewritten, flags=re.IGNORECASE):
                    rel_matches = list(re.finditer(rf"\b{relation_terms}\b", rewritten, flags=re.IGNORECASE))
                    last_rel = rel_matches[-1].group(0) if rel_matches else ""
                    if last_rel:
                        rewritten = re.sub(r"\b(she|her|he|him|they|them)\b", f"{human_target}'s {last_rel}", rewritten,
                                           flags=re.IGNORECASE)

                # General human pronoun replacement
                rewritten = re.sub(r"\b(he|she|they)\b", human_target, rewritten, flags=re.IGNORECASE)
                rewritten = re.sub(r"\b(him|them)\b", human_target, rewritten, flags=re.IGNORECASE)
                rewritten = re.sub(r"\b(his|her|their)\b", f"{human_target}'s", rewritten, flags=re.IGNORECASE)

            # --- Object Pronoun Replacement (It/Its) ---
            if non_person_target:
                # Only replace 'it'/'its' when a non-person target has been determined.
                rewritten = re.sub(r"\bit\b", non_person_target, rewritten, flags=re.IGNORECASE)
                rewritten = re.sub(r"\bits\b", f"{non_person_target}'s", rewritten, flags=re.IGNORECASE)

        return rewritten


class LLMHandler(BaseGenerator):
    # Defaulting to Qwen/Qwen2.5-7B-Instruct
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", self_check_mode: str = "balanced"):
        self.model_name = model_name
        self.self_check_mode = self_check_mode

        self.api_url = MY_API_URL
        self.api_key = os.environ.get("SILICONFLOW_API_KEY", MY_API_KEY)

        if not self.api_key:
            raise RuntimeError("Missing API key. Please set MY_API_KEY in the script.")

    def _call_llm(self, messages: List[Dict], model_name: Optional[str] = None, temperature: float = 0.1,
                  max_tokens: int = 512) -> str:
        payload = {
            "model": model_name or self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return f"Error: {resp.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate(self, query: str, retrieved_docs: List[Dict], max_new_tokens: int = 256,
                 reasoning_steps: Optional[List[str]] = None, forbid_unknown: bool = False,
                 known_facts: Optional[List[str]] = None) -> str:
        if known_facts:
            prompt = PromptBuilder.build_prompt_with_facts(query, retrieved_docs, self.model_name, reasoning_steps,
                                                           forbid_unknown, known_facts)
        else:
            prompt = PromptBuilder.build_prompt(query, retrieved_docs, self.model_name, reasoning_steps, forbid_unknown)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return self._call_llm(messages, model_name=self.model_name, temperature=0.1, max_tokens=max_new_tokens).strip()

    def plan_reasoning_steps(self, question: str) -> List[str]:
        steps = [
            "Identify the main entity (people, place, or object) mentioned in the question.",
            "Decide which attribute (location, time, relation, etc.) the question is asking about.",
            "Gather evidence from the top-ranked documents that explicitly mention this attribute.",
            "Cross-check at least two documents to confirm the consistency of the evidence."
        ]
        question_lower = question.lower()
        sub_questions = [seg.strip() for seg in re.split(r"[?]", question) if seg.strip()]
        if len(sub_questions) > 1:
            steps.append("Answer each sub-question sequentially and keep the entity references consistent.")
        if "his " in question_lower or "her " in question_lower or "their " in question_lower:
            steps.append("Resolve pronouns (his/her/their) by linking them back to the most recently mentioned entity.")
        if "why" in question_lower or "how" in question_lower:
            steps.append("Explain the causal or procedural reasoning using the evidence, not assumptions.")
        steps.append(
            "Synthesize a final answer that is concise. Prefer a supported answer over 'I don't know' whenever any evidence mentions the queried entities; only return 'I don't know' if none of the evidence is relevant.")
        return steps

    def plan_next_step(self, original_question: str, history: List[Dict]) -> Dict:
        q = original_question.strip()
        q_lower = q.lower()
        is_yes_no_original = bool(
            re.match(r"^\s*(?:is|are|was|were|do|does|did|can|could|would|should|has|have|had|will|shall)\b", q_lower))
        history_lines = []
        for i, h in enumerate(history, 1):
            sq = h.get("sub_question", "")
            ans = h.get("answer", "")
            ent = h.get("entity", "")
            history_lines.append(f"{i}. Q: {sq} | A: {ans} | Entity: {ent}")
        history_text = "\n".join(history_lines) if history_lines else "None."
        planner_prompt = (
            "You are a careful step planner for complex factual questions.\n"
            "Output ONLY a JSON object with fields: action and sub_question.\n"
            "- action: 'ask' to propose the next sub-question, or 'finalize' to stop and answer now.\n"
            "- sub_question: present ONLY when action='ask'. It MUST be a standalone factual question.\n"
            "Constraints:\n"
            "1) Do NOT invent entities not present in the original question or observed answers.\n"
            "2) Every sub-question MUST explicitly include at least one entity from the original question or observed answers (no pronouns).\n"
            "3) Prefer WH- questions (Who/What/Which/Where/When/How many/How long). Avoid yes/no questions unless the original question is yes/no.\n"
            "4) Keep sub-questions short and specific, ending with '?' and containing necessary entities explicitly.\n"
            "Original question:\n"
            f"{original_question}\n"
            "Observed steps so far:\n"
            f"{history_text}\n"
            "Your JSON:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant that outputs only JSON objects."},
            {"role": "user", "content": planner_prompt}
        ]
        raw = self._call_llm(messages, model_name=self.model_name, temperature=0.0, max_tokens=256).strip()
        cleaned = raw.strip()
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if code_block_match:
            cleaned = code_block_match.group(1).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            obj = json.loads(cleaned)
            if not isinstance(obj, dict):
                raise ValueError("Not a JSON object.")
        except Exception:
            if not history:
                return {"action": "ask", "sub_question": original_question}
            return {"action": "finalize"}

        action = str(obj.get("action", "")).strip().lower()
        sub_q = str(obj.get("sub_question", "")).strip()
        if action not in ("ask", "finalize"):
            action = "finalize" if history else "ask"
        if action == "ask":
            if not sub_q:
                sub_q = original_question
            sub_q = re.sub(r"\s+", " ", sub_q).strip()
            if not sub_q.endswith("?"):
                sub_q = sub_q.rstrip(".") + "?"

            if not is_yes_no_original and re.match(
                    r"^(?:is|are|was|were|do|does|did|can|could|would|should|has|have|had|will|shall)\b",
                    sub_q.lower()):
                try:
                    alt_subqs = self.decompose_complex_query(q)
                    alt_subqs = [s for s in alt_subqs if not re.match(
                        r"^(?:is|are|was|were|do|does|did|can|could|would|should|has|have|had|will|shall)\b",
                        s.lower())]
                    if alt_subqs:
                        sub_q = alt_subqs[0]
                except Exception:
                    pass

            entity_pool = []
            orig_ents = EntityProcessor.extract_entities(original_question)
            if orig_ents:
                entity_pool.extend(orig_ents)
            for h in history:
                he = h.get("entity", "")
                if he:
                    entity_pool.append(he)
            lower_pool = {e.lower() for e in entity_pool}
            if lower_pool and not any(e in sub_q.lower() for e in lower_pool):
                try:
                    alt_subqs = self.decompose_complex_query(q)
                    alt_subqs = self._sanitize_sub_questions(q, alt_subqs)
                    with_entities = [s for s in alt_subqs if any(e in s.lower() for e in lower_pool)]
                    if with_entities:
                        sub_q = with_entities[0]
                except Exception:
                    pass
        return {"action": action, "sub_question": sub_q}

    def _should_decompose(self, question: str) -> bool:
        q_lower = question.lower()
        triggers = [" in which ", " of which ", " who was ", " who were ", " which constituency ", " acted as ",
                    " home of ", " music director ", " first elected ", " frontman of ", " born first", " written by ",
                    " features the story ", " shot a man ", " and ", " or ", " both ", " either ", " vs ", " versus ",
                    " compared", " compare ", " are both ", " which is ", " which was ", " who is ", " that "]
        if any(t in f" {q_lower} " for t in triggers): return True
        if re.search(r"\bwas\s+.+?\s+or\s+.+?\s+born\s+first\b", q_lower): return True
        wh_count = len(re.findall(r"\b(who|which|that|where|when|whose)\b", q_lower))
        if wh_count >= 2: return True
        if wh_count >= 1 and len(q_lower) > 120: return True
        entities = EntityProcessor.extract_entities(question)
        return len(entities) >= 2

    def _rule_based_decompose(self, question: str) -> Optional[List[str]]:
        q = question.strip()
        q_sp = re.sub(r"\s+", " ", q)
        q_low = q_sp.lower()
        m_or = re.search(r"(.+?)\s+or\s+(.+?)\s+(.*)\?", q_sp)
        if m_or:
            ent1, ent2, rest = m_or.group(1).strip(), m_or.group(2).strip(), m_or.group(3).strip()
            if re.match(r"(was|is|did|does|do|can|could|should|would)", rest, re.IGNORECASE):
                rest = re.sub(r"^(was|is|did|does|do|can|could|would|should|has|have|had|will|shall)\s+", "", rest,
                              flags=re.IGNORECASE)
            return [f"{rest.capitalize()} {ent1}?", f"{rest.capitalize()} {ent2}?"]
        m_and = re.search(r"(.+?)\s+and\s+(.+?)\s+(.*)\?", q_sp)
        if m_and:
            ent1, ent2, rest = m_and.group(1).strip(), m_and.group(2).strip(), m_and.group(3).strip()
            return [f"{rest.capitalize()} {ent1}?", f"{rest.capitalize()} {ent2}?"]
        m_multi = re.search(r"(which|who|what)\s+(.+?)\s+(?:are|is|were|was)\s+(.+?)\?", q_low)
        if m_multi:
            wh, ent, attr = m_multi.group(1).capitalize(), m_multi.group(2).strip(), m_multi.group(3).strip()
            return [f"{wh} {ent} {attr}?"]
        m_clause = re.search(r"(.+?)\s+(who|which|that)\s+(.+?)\?", q_sp)
        if m_clause:
            head, rel, clause = m_clause.group(1).strip(), m_clause.group(2).strip(), m_clause.group(3).strip()
            return [f"Who is {head}?", f"What is the relation: {rel} {clause}?"]
        entities = re.split(r",| and | or |;", q_sp)
        if len(entities) > 1:
            sub_questions = []
            for ent in entities:
                ent = ent.strip()
                if ent: sub_questions.append(f"What about {ent}?")
            return sub_questions
        return None

    def decompose_complex_query(self, question: str) -> List[str]:
        if not self._should_decompose(question): return [question]
        rb = self._rule_based_decompose(question)
        if rb: return rb
        decompose_prompt = (
            f"""Your ONLY output MUST be a JSON list of strings. No other text. Example: [\"Sub-question 1?\", \"Sub-question 2?\"]
            Instructions for decomposition:
            1. If simple, return original question as ONLY item.
            2. If complex, decompose into MINIMUM sequential sub-questions.
            - Each must be direct, factual, containing AT LEAST ONE core entity.
            - DO NOT introduce new topics.
            Complex Question: {question}
            JSON Sub-questions:"""
        )
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                        {"role": "user", "content": decompose_prompt}]
            response = self._call_llm(messages, model_name=self.model_name, temperature=0.0, max_tokens=256).strip()
            cleaned = response.strip()
            code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
            if code_block_match: cleaned = code_block_match.group(1).strip()
            if cleaned.lower().startswith("json"): cleaned = cleaned[4:].strip()
            sub_queries = json.loads(cleaned)
            if not isinstance(sub_queries, list): raise ValueError("Not a list")
            sub_queries = [str(s).strip() for s in sub_queries if isinstance(s, str)]
            if not sub_queries: return [question]
            sub_queries = self._sanitize_sub_questions(question, sub_queries)
            if len(sub_queries) <= 1: return [question]
            return sub_queries
        except Exception:
            return [question]

    def self_check_answer(self, question: str, draft_answer: str, retrieved_docs: List[Dict]) -> str:
        if not draft_answer.strip(): return draft_answer.strip()
        evidence_snippets = []
        for doc in retrieved_docs[:10]:
            snippet = doc.get("content", "")
            if len(snippet) > 400: snippet = snippet[:400] + "..."
            evidence_snippets.append(f"[Doc{doc['doc_id']} score={doc['score']}] {snippet}")
        evidence_text = "\n".join(evidence_snippets) if evidence_snippets else "No evidence available."
        evidence_text_lower = evidence_text.lower()
        draft_norm = self._normalize_answer(draft_answer.splitlines()[0].strip())

        def is_supported_by_evidence(ans: str) -> bool:
            mode = getattr(self, "self_check_mode", "balanced")
            al = ans.lower().strip()
            if al in ("yes", "no"):
                if mode == "strict": return False
                ents = EntityProcessor.extract_entities(question)
                if not ents: return False
                for e in ents:
                    if e and e.lower() in evidence_text_lower: return True
                return False
            if len(al) <= 2: return False
            if al in evidence_text_lower: return True
            tokens = re.findall(r"[a-z0-9]+", al)
            tokens = [t for t in tokens if len(t) > 2]
            if not tokens: return False
            coverage_needed = 1.0 if mode == "strict" else (0.6 if mode == "balanced" else 0.4)
            hit = sum(1 for t in tokens if t in evidence_text_lower)
            threshold = max(1, math.ceil(coverage_needed * len(tokens)))
            return hit >= threshold

        verify_prompt = (
            "You are a meticulous fact-checking agent.\n"
            "Given the question, draft answer, and retrieved evidence, decide whether the answer is fully supported.\n"
            "Respond ONLY in one of the two formats below.\n"
            "APPROVED: <concise final answer>\n"
            "REVISE: <concise improved answer>\n"
            f"Question: {question}\n"
            f"Draft Answer: {draft_answer}\n"
            f"Evidence:\n{evidence_text}\n"
            "Your decision:"
        )
        messages = [{"role": "system", "content": "You are a meticulous fact-checking agent."},
                    {"role": "user", "content": verify_prompt}]
        verdict = self._call_llm(messages, model_name=self.model_name, temperature=0.0, max_tokens=256).strip()
        code_block_match = re.search(r"```(?:text|json)?\s*(.*?)\s*```", verdict, flags=re.DOTALL | re.IGNORECASE)
        if code_block_match: verdict = code_block_match.group(1).strip()

        approved_match = re.search(r"APPROVED:\s*(.+)", verdict, flags=re.IGNORECASE)
        revise_match = re.search(r"REVISE:\s*(.+)", verdict, flags=re.IGNORECASE)

        if approved_match:
            ans = self._apply_type_constraints(question, approved_match.group(1).strip())
            return ans if is_supported_by_evidence(ans) else self._apply_type_constraints(question, draft_norm)
        if revise_match:
            candidate = self._apply_type_constraints(question, revise_match.group(1).strip())
            if is_supported_by_evidence(candidate): return candidate

            def build_revise_query(orig: str, cand: str) -> str:
                toks = re.findall(r"[A-Za-z0-9]+", cand)
                toks = [t for t in toks if len(t) > 2]
                ents = EntityProcessor.extract_entities(cand)
                added = []
                for s in ents + toks:
                    s_clean = s.strip()
                    if s_clean and s_clean.lower() not in added: added.append(s_clean.lower())
                if added: return f"{orig} {' '.join(sorted(set(added)))}"
                return orig

            refined_query = build_revise_query(question, candidate)
            try:
                new_docs = retriever.retrieve(refined_query, top_k=10)
            except Exception:
                return draft_norm
            try:
                new_reranked = DocProcessor.rerank_docs(refined_query, new_docs)
            except Exception:
                new_reranked = new_docs
            new_pruned = DocProcessor.prune_context_documents(new_reranked, max_doc=10)

            entities = EntityProcessor.extract_entities(refined_query)
            coverage = {e.lower(): 0 for e in entities}
            for d in new_pruned:
                cl = d.get("content", "").lower()
                for e in list(coverage.keys()):
                    if e and e in cl: coverage[e] += 1
            has_any_coverage = any(c > 0 for c in coverage.values())

            reasoning_steps = self.plan_reasoning_steps(refined_query)
            new_answer = self.generate(refined_query, new_pruned, reasoning_steps=reasoning_steps,
                                       forbid_unknown=has_any_coverage)
            new_answer_norm = self._apply_type_constraints(refined_query, new_answer.splitlines()[0].strip())

            new_ev_snippets = []
            for doc in new_pruned[:10]:
                snippet = doc.get("content", "")
                if len(snippet) > 400: snippet = snippet[:400] + "..."
                new_ev_snippets.append(f"[Doc{doc['doc_id']} score={doc['score']}] {snippet}")
            new_ev_text_lower = "\n".join(new_ev_snippets).lower() if new_ev_snippets else ""

            def is_supported_new(ans: str) -> bool:
                mode = getattr(self, "self_check_mode", "balanced")
                al = ans.lower().strip()
                if al in ("yes", "no"):
                    if mode == "strict": return False
                    ents = EntityProcessor.extract_entities(refined_query)
                    for e in ents:
                        if e and e.lower() in new_ev_text_lower: return True
                    return False
                if len(al) <= 2: return False
                if al in new_ev_text_lower: return True
                toks = re.findall(r"[a-z0-9]+", al)
                toks = [t for t in toks if len(t) > 2]
                if not toks: return False
                coverage_needed = 1.0 if mode == "strict" else (0.6 if mode == "balanced" else 0.4)
                hit = sum(1 for t in toks if t in new_ev_text_lower)
                threshold = max(1, math.ceil(coverage_needed * len(toks)))
                return hit >= threshold

            print(f"DEBUG: REVISE → re-retrieval. Candidate: {candidate}; New: {new_answer_norm}")
            return new_answer_norm if is_supported_new(new_answer_norm) else draft_norm

        return self._apply_type_constraints(question, draft_norm)

    @staticmethod
    def _normalize_answer(ans: str) -> str:
        a = ans.strip()
        a = re.sub(r'^(?:final answer:)\s*', '', a, flags=re.IGNORECASE).strip()
        low = a.lower()
        if low.startswith("yes"): return "Yes"
        if low.startswith("no"): return "No"
        if len(a) >= 2 and ((a[0] == '"' and a[-1] == '"') or (a[0] == "'" and a[-1] == "'")):
            a = a[1:-1].strip()
        a = re.sub(r'[\.;:!?]+$', '', a).strip()
        a = re.sub(r'\s+', ' ', a)
        return a

    def _apply_type_constraints(self, question: str, ans: str) -> str:
        ql = question.lower()
        a = self._normalize_answer(ans)
        if re.search(r'^\s*(?:are|is|do|does|did|can)\b', ql) or (" both " in f" {ql} ") or (" either " in f" {ql} "):
            if a.lower().startswith('yes'): return 'Yes'
            if a.lower().startswith('no'): return 'No'
            return a.split(',')[0].split(';')[0].strip()
        if 'when' in ql:
            patterns = [
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',
                r'\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b',
                r'\b(?:in|on)\s+(1[5-9]\d{2}|20\d{2}|21\d{2})\b'
            ]
            for pat in patterns:
                m = re.search(pat, ans, flags=re.IGNORECASE)
                if m: return self._normalize_answer(m.group(0))
            return ' '.join(a.split()[:7])
        if 'where' in ql:
            ents = EntityProcessor.extract_entities(ans)
            if ents: return self._normalize_answer(max(ents, key=len))
            return ' '.join(a.split()[:7])
        if 'who' in ql:
            ents = EntityProcessor.extract_entities(ans)
            if ents: return self._normalize_answer(max(ents, key=len))
            return ' '.join(a.split()[:5])
        if 'what' in ql:
            a2 = re.split(r'[;,\n]', a)[0]
            return ' '.join(a2.split()[:10])
        return a

    def _sanitize_sub_questions(self, original_question: str, sub_questions: List[str]) -> List[str]:
        q_lower = original_question.lower()
        cleaned = [re.sub(r"\s+", " ", s.strip()) for s in sub_questions if isinstance(s, str) and s.strip()]
        is_yes_no_original = bool(
            re.match(r"^\s*(?:is|are|was|were|do|does|did|can|could|would|should|has|have|had|will|shall)\b", q_lower))
        if not is_yes_no_original:
            cleaned = [s for s in cleaned if not re.match(
                r"^(?:is|are|was|were|do|does|did|can|could|would|should|has|have|had|will|shall)\b", s.lower())]
        original_entities = EntityProcessor.extract_entities(original_question)
        if original_entities:
            lower_entities = [e.lower() for e in original_entities]
            kept = [s for s in cleaned if any(e in s.lower() for e in lower_entities)]
        else:
            kept = cleaned
        seen = set()
        result = []
        for s in kept:
            norm = re.sub(r"\s+", " ", s.lower()).strip(" ?.!.,")
            if norm in seen: continue
            seen.add(norm)
            if not s.endswith("?"): s = s.rstrip(".") + "?"
            result.append(s)
        return result or [original_question]


class RAGPipelineManager:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.entity_processor = EntityProcessor()
        self.prompt_builder = PromptBuilder()
        self.doc_processor = DocProcessor()
        self.multi_turn_manager = MultiTurnManager()
        self.llm_handler = LLMHandler(model_name)

    def _run_iterative_planner(self, session_id: str, original_question: str, retrieval_method):
        if retriever is None: raise RuntimeError("Retriever not available.")
        history: List[Dict] = []
        doc_by_id: Dict = {}
        known_facts: List[str] = []
        asked_norm = set()
        max_steps = 4

        for step in range(max_steps):
            plan = self.llm_handler.plan_next_step(original_question, history)
            act = plan.get("action", "").lower()
            if act == "finalize": break
            if act != "ask": break
            subq = plan.get("sub_question", "").strip()

            norm_subq = re.sub(r"\s+", " ", subq.lower()).strip(" ?.!.,")
            if norm_subq in asked_norm:
                print(f"[Plan] Duplicate sub-question detected, stopping: {subq}")
                break
            asked_norm.add(norm_subq)
            if not subq.endswith("?"): subq = subq.rstrip(".") + "?"
            print(f"[Plan] Step {step + 1} sub-question: {subq}")

            try:
                docs = retriever.retrieve(subq, top_k=10)
            except Exception as e:
                print(f"DEBUG: retrieval failed for subq '{subq}': {e}")
                break

            try:
                reranked = self.doc_processor.rerank_docs(subq, docs)
            except Exception:
                reranked = docs
            pruned = self.doc_processor.prune_context_documents(reranked, max_doc=10)

            sub_reasoning = self.llm_handler.plan_reasoning_steps(subq)
            sub_ans = self.llm_handler.generate(subq, pruned, reasoning_steps=sub_reasoning, forbid_unknown=False)
            sub_ans_norm = self.llm_handler._apply_type_constraints(subq, sub_ans.splitlines()[0].strip())
            ent = self.entity_processor.extract_main_entity(subq, sub_ans_norm)
            history.append({"sub_question": subq, "answer": sub_ans_norm, "entity": ent, "doc_count": len(pruned)})
            known_facts.append(f"{subq} -> {sub_ans_norm}")
            print(f"[Observe] Step {step + 1} answer: {sub_ans_norm}; entity: {ent}")

            for d in pruned:
                di = d.get("doc_id")
                prev = doc_by_id.get(di)
                if prev is None or d.get("score", 0) > prev.get("score", 0):
                    doc_by_id[di] = d

        aggregated_docs = list(doc_by_id.values())
        if not aggregated_docs:
            aggregated_docs = retriever.retrieve(original_question, top_k=10)
        pruned_docs = self.doc_processor.prune_context_documents(aggregated_docs, max_doc=10)

        entities = EntityProcessor.extract_entities(original_question)
        coverage = {e.lower(): 0 for e in entities}
        for d in pruned_docs:
            cl = d.get("content", "").lower()
            for e in list(coverage.keys()):
                if e and e in cl: coverage[e] += 1
        has_any_coverage = any(c > 0 for c in coverage.values())

        print("[Update] Known facts:")
        for i, f in enumerate(known_facts, 1): print(f"  {i}. {f}")

        final_reasoning = self.llm_handler.plan_reasoning_steps(original_question)
        final_answer = self.llm_handler.generate(original_question, pruned_docs, reasoning_steps=final_reasoning,
                                                 forbid_unknown=has_any_coverage, known_facts=known_facts)
        final_answer = self.llm_handler.self_check_answer(original_question, final_answer, pruned_docs)

        ans_for_entity = None if final_answer.lower().startswith("i don't know") else final_answer
        question_entity = self.entity_processor.extract_main_entity(original_question, answer=None)
        answer_entity = self.entity_processor.extract_main_entity("", ans_for_entity) if ans_for_entity else ""
        main_entity = question_entity or answer_entity or ""

        self.multi_turn_manager.multi_turn_contexts[session_id].append({
            "question": original_question,
            "rewritten_query": original_question,
            "answer": final_answer,
            "main_entity": main_entity,
            "question_entity": question_entity,
            "answer_entity": answer_entity,
            "plan_history": history
        })
        formatted_retrieved_docs = [[doc['doc_id'], doc['score']] for doc in pruned_docs]
        return {
            "final_answer": final_answer,
            "formatted_retrieved_docs": formatted_retrieved_docs,
            "rewritten_query": original_question,
            "plan_history": history,
            "known_facts": known_facts
        }

    def run(self, session_id: str, question: str, retrieval_method: str = "dense") -> Dict:
        self.multi_turn_manager.init_multi_turn_context(session_id)
        rewritten_query = self.multi_turn_manager.rewrite_query(session_id, question)
        print(f"Rewritten Query: {rewritten_query} | Retrieval Method: {retrieval_method}")
        try:
            result = self._run_iterative_planner(session_id, rewritten_query, retrieval_method)
            return result
        except Exception as e:
            print(f"DEBUG: iterative planner failed or skipped: {e}. Falling back to static pipeline.")

        if retriever is None:
            print("DEBUG: No retriever available; answering with LLM only.")
            reasoning_steps = self.llm_handler.plan_reasoning_steps(rewritten_query)
            draft_answer = self.llm_handler.generate(rewritten_query, [], reasoning_steps=reasoning_steps,
                                                     forbid_unknown=False)
            final_answer = self.llm_handler._apply_type_constraints(rewritten_query,
                                                                    draft_answer.splitlines()[0].strip())
            return {
                "final_answer": final_answer,
                "formatted_retrieved_docs": [],
                "rewritten_query": rewritten_query,
                "plan_history": [],
                "known_facts": []
            }

        sub_queries = self.llm_handler.decompose_complex_query(rewritten_query)
        print(f"Decomposed Sub-queries: {sub_queries}")
        all_retrieved_docs = []
        for sub_query in sub_queries:
            try:
                docs_for_sub_query = retriever.retrieve(sub_query, top_k=10)
            except Exception as e:
                docs_for_sub_query = []
            all_retrieved_docs.extend(docs_for_sub_query)

        doc_by_id = {}
        for d in all_retrieved_docs:
            di = d.get("doc_id")
            prev = doc_by_id.get(di)
            if prev is None or d.get("score", 0) > prev.get("score", 0):
                doc_by_id[di] = d
        retrieved_docs = list(doc_by_id.values())

        try:
            reranked_docs = self.doc_processor.rerank_docs_by_subqueries(rewritten_query, sub_queries, retrieved_docs)
        except Exception:
            reranked_docs = retrieved_docs
        pruned_docs = self.doc_processor.prune_context_documents(reranked_docs, max_doc=10)

        entities = EntityProcessor.extract_entities(rewritten_query)
        coverage = {e.lower(): 0 for e in entities}
        for d in pruned_docs:
            cl = d.get("content", "").lower()
            for e in list(coverage.keys()):
                if e and e in cl: coverage[e] += 1

        missing = [e for e, c in coverage.items() if c == 0]
        if missing:
            print(f"DEBUG: Fallback retrieval for missing entities: {missing}")
            extra_docs = []
            for e in missing:
                try:
                    extra_docs.extend(retriever.retrieve(e, top_k=10))
                except Exception:
                    pass
            for ed in extra_docs:
                di = ed.get("doc_id")
                prev = doc_by_id.get(di)
                if prev is None or ed.get("score", 0) > prev.get("score", 0):
                    doc_by_id[di] = ed
            retrieved_docs = list(doc_by_id.values())
            pruned_docs = self.doc_processor.prune_context_documents(retrieved_docs, max_doc=10)

            coverage = {e.lower(): 0 for e in entities}
            for d in pruned_docs:
                cl = d.get("content", "").lower()
                for e in list(coverage.keys()):
                    if e and e in cl: coverage[e] += 1

        print("Pruned Documents (top 3 content snippets):\n" + "\n".join(
            [f"[Doc{d['doc_id']}] {d['content'][:150]}..." for d in pruned_docs[:3]]))

        reasoning_steps = self.llm_handler.plan_reasoning_steps(rewritten_query)
        has_any_coverage = any(c > 0 for c in coverage.values())
        draft_answer = self.llm_handler.generate(rewritten_query, pruned_docs, reasoning_steps=reasoning_steps,
                                                 forbid_unknown=has_any_coverage)
        final_answer = self.llm_handler.self_check_answer(rewritten_query, draft_answer, pruned_docs)

        ans_for_entity = None if final_answer.lower().startswith("i don't know") else final_answer
        question_entity = self.entity_processor.extract_main_entity(rewritten_query, answer=None)
        answer_entity = self.entity_processor.extract_main_entity("", ans_for_entity) if ans_for_entity else ""
        main_entity = question_entity or answer_entity or ""

        self.multi_turn_manager.multi_turn_contexts[session_id].append({
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": final_answer,
            "main_entity": main_entity,
            "question_entity": question_entity,
            "answer_entity": answer_entity
        })
        formatted_retrieved_docs = [[doc['doc_id'], doc['score']] for doc in pruned_docs]
        return {
            "final_answer": final_answer,
            "formatted_retrieved_docs": formatted_retrieved_docs,
            "rewritten_query": rewritten_query,
            "plan_history": [],
            "known_facts": [],
            "reasoning_steps": reasoning_steps
        }


# 3. Main Program Entry
if __name__ == '__main__':
    # Configure input/output file list
    # Assuming retrieval.py downloaded files into data/ directory
    tasks = [
        {
            "input": "data/validation.jsonl",
            "output": "val_prediction.jsonl",
            "desc": "Validation Set (Self-check)"
        },
        {
            "input": "data/test.jsonl",
            "output": "test_prediction.jsonl",
            "desc": "Test Set (Final Submission)"
        }
    ]

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f">>> Initializing RAG Pipeline with model: {model_name}")
    rag_pipeline = RAGPipelineManager(model_name=model_name)

    session_id = "batch_runner"

    for task in tasks:
        input_file = task["input"]
        output_file = task["output"]
        description = task["desc"]

        if not os.path.exists(input_file):
            print(f"\nSkipping {description}: File {input_file} not found.")
            continue

        print(f"\n=== Processing {description} ===")
        print(f"Reading from: {input_file}")

        with open(input_file, 'r', encoding='utf8') as fin:
            data = [json.loads(l) for l in fin]

        results = []
        # Use tqdm to display progress bar
        for turn_id, item in enumerate(tqdm(data, desc=f"Generating {description}")):
            user_query = item["text"] if "text" in item else item.get("query")
            # print(f"Processing: {user_query}")

            try:
                # Run Pipeline
                res_dict = rag_pipeline.run(session_id, user_query)
                final_answer = res_dict.get("final_answer", "")
                retrieved_docs = res_dict.get("formatted_retrieved_docs", [])

                # Construct output result
                result_item = {
                    "id": item.get("id", f"{session_id}_turn{turn_id + 1}"),
                    "question": user_query,
                    "answer": final_answer,
                    "retrieved_docs": retrieved_docs
                }
                results.append(result_item)
            except Exception as e:
                print(f"Error processing id={item.get('id')}: {e}")
                # Write placeholder on error to maintain line count integrity
                results.append({
                    "id": item.get("id", ""),
                    "question": user_query,
                    "answer": "Error",
                    "retrieved_docs": []
                })

        # Write results to output file
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w', encoding='utf8') as fout:
            for res in results:
                fout.write(json.dumps(res, ensure_ascii=False) + '\n')

    print("\nAll tasks completed.")