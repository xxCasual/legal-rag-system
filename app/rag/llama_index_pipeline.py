"""LlamaIndex-based production retrieval pipeline."""

from __future__ import annotations

import queue
import threading
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.core.config import settings


VALID_CRAG_MODES = {"llm", "reranker", "off"}


def normalize_crag_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in VALID_CRAG_MODES:
        choices = ", ".join(sorted(VALID_CRAG_MODES))
        raise ValueError(f"LEGAL_RAG_CRAG_MODE must be one of: {choices}")
    return normalized


class LlamaIndexLegalRAGPipeline:
    """Legal RAG pipeline backed by LlamaIndex retrievers and postprocessors."""

    LAW_COLLECTION = "legal_law_docs"

    def __init__(
        self,
        data_dir: str = str(settings.data_dir),
        persist_dir: str = str(settings.llama_law_chroma_persist_dir),
        embedding_model: str = settings.embedding_model,
        reranker_model: str = settings.reranker_model,
        llm_model: str = settings.llm_model,
        bm25_k: int = 8,
        vector_k: int = 8,
        rerank_top_n: int = 4,
        rrf_k_const: int = 60,
        crag_mode: str = settings.crag_mode,
        rebuild_index: bool = False,
        verbose: bool = True,
        llm_timeout_seconds: float = settings.llm_timeout_seconds,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.llm_model = llm_model
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.rerank_top_n = rerank_top_n
        self.rrf_k_const = rrf_k_const
        self.crag_mode = normalize_crag_mode(crag_mode)
        self.verbose = verbose
        self.llm_timeout_seconds = llm_timeout_seconds

        api_key = os.environ.get("DEEPSEEK_API_KEY") or settings.deepseek_api_key
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        self.api_key = api_key

        self._configure_llama_index()
        self.nodes = self._load_nodes()
        self.index = self._build_or_load_index(rebuild_index=rebuild_index)
        self.vector_retriever = self.index.as_retriever(similarity_top_k=vector_k)
        self.bm25_retriever = self._build_bm25_retriever(self.nodes)
        self.reranker = self._build_reranker()

    def query(self, question: str) -> Tuple[str, List[str]]:
        result = self.query_with_details(question)
        return result["answer"], result["contexts"]

    def query_with_details(self, question: str) -> Dict[str, Any]:
        started_at = time.perf_counter()
        route = self._route(question)

        if "知识库外" in route or "无关" in route:
            answer = self._reject(question)
            return {
                "route": route,
                "answer": answer,
                "contexts": [],
                "chunks_retrieved": 0,
                "chunks_after_filter": 0,
                "crag_mode": self.crag_mode,
                "latency": round(time.perf_counter() - started_at, 3),
            }

        retrieved = (
            self._hybrid_rag_fusion_then_rerank(question)
            if "咨询" in route
            else self._hybrid_retrieve_then_rerank(question)
        )
        filtered = self._crag_filter(question, retrieved)
        used_nodes = filtered[:4]
        contexts = [self._node_text(item) for item in used_nodes]
        answer = self._answer(question, contexts)

        return {
            "route": route,
            "answer": answer,
            "contexts": contexts,
            "chunks_retrieved": len(retrieved),
            "chunks_after_filter": len(filtered),
            "crag_mode": self.crag_mode,
            "latency": round(time.perf_counter() - started_at, 3),
        }

    def _configure_llama_index(self) -> None:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai_like import OpenAILike

        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        Settings.llm = OpenAILike(
            model=self.llm_model,
            api_key=self.api_key,
            api_base=settings.deepseek_base_url,
            is_chat_model=True,
            timeout=30,
            max_retries=0,
        )
        self.llm = Settings.llm

    def _load_nodes(self) -> List[Any]:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter

        self._log(f"📚 LlamaIndex 加载法律文档: {self.data_dir}")
        documents = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            required_exts=[".txt"],
        ).load_data()
        splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=300)
        nodes = splitter.get_nodes_from_documents(documents)
        self._log(f"   {len(documents)} 文档, {len(nodes)} nodes")
        return nodes

    def _build_or_load_index(self, rebuild_index: bool) -> Any:
        import chromadb
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        if rebuild_index and self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_or_create_collection(self.LAW_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if collection.count() == 0 or rebuild_index:
            self._log("✅ 构建 LlamaIndex 法律向量索引")
            return VectorStoreIndex(
                self.nodes,
                storage_context=storage_context,
                show_progress=self.verbose,
            )
        self._log(f"✅ 加载 LlamaIndex 法律向量索引: {self.persist_dir}")
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    def _build_bm25_retriever(self, nodes: List[Any]) -> Any:
        from llama_index.retrievers.bm25 import BM25Retriever

        try:
            return BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=self.bm25_k,
                tokenizer=self._chinese_tokenize,
            )
        except TypeError:
            return BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=self.bm25_k,
            )

    def _build_reranker(self) -> Any:
        from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

        return SentenceTransformerRerank(
            model=self.reranker_model,
            top_n=self.rerank_top_n,
        )

    def _route(self, question: str) -> str:
        heuristic_route = self._heuristic_route(question)
        if heuristic_route is not None:
            return heuristic_route

        prompt = """你是一个法律问题分类器。本系统的知识库包含 4 部中国法律：
《劳动法》、《劳动合同法》、《劳动争议调解仲裁法》、《保险法》。

请判断用户问题属于以下哪个类型，只回答类型名称：
- "法条查询"：询问具体的法律条文规定
- "法律咨询"：描述具体情境寻求建议
- "知识库外"：与上述 4 部法律完全无关

注意：涉及劳动法、劳动合同、工资、加班、解雇、工伤、仲裁、保险、理赔的问题都属于知识库内。

用户问题：{question}
类型：""".format(question=question)
        return self._safe_complete(prompt, fallback="知识库外")

    def _reject(self, question: str) -> str:
        prompt = """用户问了一个与中国劳动法律无关的问题。请礼貌地告诉用户这个系统只能回答劳动法相关的问题，
并给出1-2个它能回答的问题示例。
用户问题：{question}""".format(question=question)
        return self._safe_complete(prompt, fallback="该问题超出当前劳动法律知识库范围。")

    def _answer(self, question: str, contexts: List[str]) -> str:
        context = "\n\n".join(contexts)
        prompt = """你是一个专业的中国法律顾问。请严格根据以下法律条文来回答用户的问题。

要求：
1. 只根据提供的法律条文回答，不要使用你自己的知识
2. 回答中请引用具体的法律名称和条款编号
3. 如果提供的条文中没有相关信息，请明确说"根据现有资料无法回答该问题"
4. 用通俗易懂的语言解释

参考法律条文：
{context}

用户问题：{question}""".format(context=context, question=question)
        fallback = self._fallback_answer(contexts)
        return self._safe_complete(prompt, fallback=fallback)

    def _generate_queries(self, question: str) -> List[str]:
        prompt = """你是一个法律检索助手。请把用户的法律问题改写成3个不同角度的版本，
用于在法律法规数据库中检索。每个版本一行，不要编号，不要额外说明。
原始问题：{question}""".format(question=question)
        text = self._safe_complete(prompt, fallback="")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _hybrid_retrieve_then_rerank(self, query: str) -> List[Any]:
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query)
        fused = self._rrf_fuse([bm25_nodes, vector_nodes], self.rrf_k_const)
        return self._rerank(query, fused)

    def _hybrid_rag_fusion_then_rerank(self, question: str) -> List[Any]:
        per_query_results = []
        for query in [question] + self._generate_queries(question):
            bm25_nodes = self.bm25_retriever.retrieve(query)
            vector_nodes = self.vector_retriever.retrieve(query)
            per_query_results.append(
                self._rrf_fuse([bm25_nodes, vector_nodes], self.rrf_k_const)
            )
        fused = self._rrf_fuse(per_query_results, self.rrf_k_const)
        return self._rerank(question, fused)

    def _rerank(self, query: str, candidates: List[Any]) -> List[Any]:
        if not candidates:
            return []
        if self.reranker is None:
            return candidates[: self.rerank_top_n]
        try:
            return self.reranker.postprocess_nodes(candidates, query_str=query)[
                : self.rerank_top_n
            ]
        except TypeError:
            from llama_index.core import QueryBundle

            return self.reranker.postprocess_nodes(
                candidates,
                query_bundle=QueryBundle(query),
            )[: self.rerank_top_n]

    def _crag_filter(self, question: str, nodes: List[Any]) -> List[Any]:
        if self.crag_mode in {"off", "reranker"}:
            return nodes

        relevant = []
        for node in nodes:
            prompt = """判断以下法律条文是否与用户问题相关。只回答"yes"或"no"。
法律条文：{document}
用户问题：{question}""".format(document=self._node_text(node), question=question)
            score = self._safe_complete(prompt, fallback="yes").strip().lower()
            if score == "yes":
                relevant.append(node)
        return relevant or nodes[:2]

    def _heuristic_route(self, question: str) -> str | None:
        out_of_scope_hints = (
            "天气",
            "股票",
            "Python",
            "代码",
            "破解",
            "密码",
            "菜谱",
            "旅游",
            "电影",
        )
        if any(hint in question for hint in out_of_scope_hints):
            return "知识库外"

        law_hints = (
            "劳动",
            "劳动合同",
            "合同",
            "工资",
            "加班",
            "试用期",
            "社保",
            "社会保险",
            "工伤",
            "仲裁",
            "辞退",
            "解雇",
            "解除",
            "赔偿",
            "补偿",
            "休假",
            "年假",
            "保险",
            "理赔",
        )
        if not any(hint in question for hint in law_hints):
            return None

        consult_hints = (
            "怎么办",
            "能不能",
            "可以",
            "是否合法",
            "是否",
            "如果",
            "该怎么",
            "如何处理",
            "怎么处理",
            "赔偿",
            "补偿",
            "拒绝",
        )
        return "法律咨询" if any(hint in question for hint in consult_hints) else "法条查询"

    def _safe_complete(self, prompt: str, fallback: str) -> str:
        try:
            return self._complete(prompt).strip()
        except Exception as exc:
            self._log(f"⚠️ LLM 调用失败，使用降级结果: {type(exc).__name__}: {exc}")
            return fallback

    def _fallback_answer(self, contexts: List[str]) -> str:
        if not contexts:
            return "当前检索结果为空，请补充问题细节或更新法律知识库后再试。"
        preview = "\n\n".join(contexts[:2])
        return (
            "根据检索到的法律条文，相关依据包括：\n"
            f"{preview}\n\n"
            "请结合完整条文和具体事实进行人工复核。"
        )

    @staticmethod
    def _rrf_fuse(results_lists: Iterable[List[Any]], k_const: int) -> List[Any]:
        scores: Dict[str, float] = {}
        node_map: Dict[str, Any] = {}
        for result in results_lists:
            for rank, item in enumerate(result, start=1):
                key = LlamaIndexLegalRAGPipeline._node_text(item)
                if key not in scores:
                    scores[key] = 0.0
                    node_map[key] = item
                scores[key] += 1.0 / (k_const + rank)
        return [node_map[key] for key in sorted(scores, key=scores.get, reverse=True)]

    @staticmethod
    def _node_text(item: Any) -> str:
        node = getattr(item, "node", item)
        if hasattr(node, "get_content"):
            return str(node.get_content())
        return str(getattr(node, "text", getattr(node, "page_content", "")))

    @staticmethod
    def _chinese_tokenize(text: str) -> List[str]:
        import re

        pattern = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]|[^\s]")
        return [match.group() for match in pattern.finditer(text)]

    def _complete(self, prompt: str) -> str:
        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def run_completion() -> None:
            try:
                response_queue.put(("ok", self.llm.complete(prompt)))
            except Exception as exc:  # pragma: no cover - exercised via _safe_complete
                response_queue.put(("error", exc))

        seconds = getattr(self, "llm_timeout_seconds", 30)
        if seconds <= 0:
            response = self.llm.complete(prompt)
            return str(getattr(response, "text", response))

        worker = threading.Thread(target=run_completion, daemon=True)
        worker.start()
        worker.join(seconds)
        if worker.is_alive():
            raise TimeoutError(f"LLM call exceeded {seconds}s")

        status, payload = response_queue.get_nowait()
        if status == "error":
            raise payload
        return str(getattr(payload, "text", payload))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
