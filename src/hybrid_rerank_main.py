"""
src/hybrid_rerank_main.py
==========================
Hybrid 检索 + Reranker 重排序版法律 RAG。

与 hybrid_main.py 的唯一区别：
- 把 retriever_k 从 4 调到 8（给 reranker 更多原料）
- 在 _hybrid_retrieve / _hybrid_rag_fusion_retrieve 后追加 reranker 重排
- 重排后取 top-N（默认 4）作为最终 contexts

设计动机：
- hybrid 评估发现 Context Precision 0.80（不错但仍有 10/48 样本 < 0.5）
- 这部分样本是"召回对了但排序里夹杂无关 chunk"，正是 reranker 的甜蜜点
- BGE-reranker-base 用 cross-encoder 直接判断 (query, doc) 相关性，
  比 embedding 余弦相似度精度高得多
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class HybridRerankLegalRAGPipeline:
    """Hybrid 检索 + BGE Reranker 重排序版法律 RAG。"""

    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        llm_model: str = "deepseek-chat",
        # 召回数：扩大到 8（hybrid 时是 4），给 reranker 更多原料
        bm25_k: int = 8,
        vector_k: int = 8,
        # 重排后保留 top-N 作为最终 contexts
        rerank_top_n: int = 4,
        rrf_k_const: int = 60,
        rebuild_index: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            bm25_k / vector_k: 各自召回多少 chunk（融合前）
            rerank_top_n: reranker 排序后保留多少 chunk（送进 CRAG）
            reranker_model: HuggingFace 上的 reranker 模型名
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.rerank_top_n = rerank_top_n
        self.rrf_k_const = rrf_k_const
        self.verbose = verbose

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        # ---- 文档 + 切分 ----
        self._log(f"📚 加载文档: {self.data_dir}")
        loader = DirectoryLoader(
            str(self.data_dir), glob="*.txt",
            loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300,
            separators=["\n\n", "\n", "。", " ", ""],
        )
        self.splits: List[Document] = splitter.split_documents(docs)
        self._log(f"   {len(docs)} 文档, {len(self.splits)} chunks")

        # ---- 向量索引 ----
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        persist_path = Path(persist_dir)
        index_exists = persist_path.exists() and any(persist_path.iterdir())

        if rebuild_index or not index_exists:
            if persist_path.exists():
                shutil.rmtree(persist_dir)
            self.vectorstore = Chroma.from_documents(
                self.splits, embeddings, persist_directory=persist_dir
            )
            self._log("✅ 向量索引已构建")
        else:
            self.vectorstore = Chroma(
                persist_directory=persist_dir, embedding_function=embeddings
            )
            self._log(f"✅ 向量索引已加载: {persist_dir}")

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": vector_k}
        )

        # ---- BM25 ----
        self.bm25_retriever = self._build_bm25_retriever(self.splits, k=bm25_k)
        self._log("✅ BM25 索引已构建")

        # ---- Reranker（关键的新组件）----
        self._log(f"📦 加载 reranker: {reranker_model}")
        self.reranker = self._load_reranker(reranker_model)
        self._log("✅ Reranker 已加载")

        # ---- LLM ----
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        self._build_chains()

    # ----------------------------------------------------------------
    # Reranker 加载（用 sentence-transformers 的 CrossEncoder）
    # ----------------------------------------------------------------

    @staticmethod
    def _load_reranker(model_name: str):
        """
        用 sentence-transformers 的 CrossEncoder 加载。
        CrossEncoder 接受 [(q, d1), (q, d2), ...] 列表，输出每对的相关性分数。
        """
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CrossEncoder(model_name, device=device, max_length=512)

    # ----------------------------------------------------------------
    # 重排：Hybrid 候选 → Reranker 打分 → 按分数降序取 top_n
    # ----------------------------------------------------------------

    def _rerank(
        self, query: str, candidates: List[Document], top_n: int
    ) -> List[Document]:
        """
        对 candidates 做 reranker 重排，返回分数最高的 top_n 个。
        """
        if not candidates:
            return []
        if len(candidates) <= top_n:
            # 候选数小于等于 top_n，但仍重排（保证顺序最优）
            top_n = len(candidates)

        # 构造 (query, doc) pairs
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        # 按分数降序排序
        indexed = list(zip(candidates, scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if self.verbose:
            self._log("  🎯 Reranker 重排前后对比:")
            for rank, (doc, score) in enumerate(indexed[:top_n], 1):
                preview = doc.page_content[:50].replace("\n", " ")
                self._log(f"     #{rank} score={score:.4f} | {preview}...")

        return [doc for doc, _ in indexed[:top_n]]

    # ----------------------------------------------------------------
    # 工具
    # ----------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _chinese_tokenize(text: str) -> List[str]:
        import re
        tokens = []
        pattern = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]|[^\s]")
        for m in pattern.finditer(text):
            tokens.append(m.group())
        return tokens

    def _build_bm25_retriever(self, documents: List[Document], k: int):
        retriever = BM25Retriever.from_documents(
            documents, preprocess_func=self._chinese_tokenize,
        )
        retriever.k = k
        return retriever

    @staticmethod
    def _rrf_fuse(
        results_lists: List[List[Document]], k_const: int = 60
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for result in results_lists:
            for rank, doc in enumerate(result, start=1):
                key = doc.page_content
                if key not in scores:
                    scores[key] = 0.0
                    doc_map[key] = doc
                scores[key] += 1.0 / (k_const + rank)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    # ----------------------------------------------------------------
    # 检索（hybrid + rerank）
    # ----------------------------------------------------------------

    def _hybrid_retrieve_then_rerank(self, query: str) -> List[Document]:
        """
        单 query 流程：
        1. BM25 召回 bm25_k 个 + 向量召回 vector_k 个
        2. RRF 融合（去重）
        3. Reranker 重排，取 top rerank_top_n
        """
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        fused = self._rrf_fuse([bm25_docs, vector_docs], k_const=self.rrf_k_const)
        # fused 此时可能有 8-16 个候选（去重后），交给 reranker 选 top_n
        return self._rerank(query, fused, self.rerank_top_n)

    def _hybrid_rag_fusion_then_rerank(self, question: str) -> List[Document]:
        """
        多 query 流程：
        1. 用 LLM 生成 3 个 query 改写
        2. 每个 query 各自做 hybrid 召回
        3. 全部结果再 RRF 融合
        4. Reranker 重排，取 top rerank_top_n
        """
        queries = self.generate_queries.invoke({"question": question})
        self._log("  📝 RAG-Fusion 改写：")
        for i, q in enumerate(queries):
            self._log(f"     Q{i + 1}: {q}")

        all_queries = [question] + queries
        per_query_results = []
        for q in all_queries:
            bm25_docs = self.bm25_retriever.invoke(q)
            vector_docs = self.vector_retriever.invoke(q)
            fused_q = self._rrf_fuse([bm25_docs, vector_docs], k_const=self.rrf_k_const)
            per_query_results.append(fused_q[: max(self.bm25_k, self.vector_k)])

        # 跨 query 再融合
        all_fused = self._rrf_fuse(per_query_results, k_const=self.rrf_k_const)

        # Reranker 重排（用原始 question，不是改写后的）
        return self._rerank(question, all_fused, self.rerank_top_n)

    # ----------------------------------------------------------------
    # Chains
    # ----------------------------------------------------------------

    def _build_chains(self):
        router_prompt = ChatPromptTemplate.from_template(
            """你是一个法律问题分类器。本系统的知识库包含 4 部中国法律：
《劳动法》、《劳动合同法》、《劳动争议调解仲裁法》、《保险法》。

请判断用户问题属于以下哪个类型，只回答类型名称：
- "法条查询"：询问具体的法律条文规定
- "法律咨询"：描述具体情境寻求建议
- "知识库外"：与上述 4 部法律完全无关（如天气、编程、医疗、刑事案件）

注意：
- 涉及"劳动法 / 劳动合同 / 工资 / 加班 / 解雇 / 工伤 / 仲裁 / 保险 / 理赔"等的问题都属于知识库内
- 即使是抽象的政策、原则类问题（如"劳动法如何保护劳动者"）也属于"法条查询"
- 跨法律的对比问题（如"保险法和劳动法在XX上有何不同"）属于"法条查询"

用户问题：{question}
类型："""
        )
        self.route_chain = (
            router_prompt | self.llm | StrOutputParser() | (lambda x: x.strip())
        )

        multi_query_prompt = ChatPromptTemplate.from_template(
            """你是一个法律检索助手。请把用户的法律问题改写成3个不同角度的版本，
用于在法律法规数据库中检索。每个版本一行，不要编号，不要额外说明。
原始问题：{question}"""
        )
        self.generate_queries = (
            multi_query_prompt | self.llm | StrOutputParser()
            | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()])
        )

        grader_prompt = ChatPromptTemplate.from_template(
            """判断以下法律条文是否与用户问题相关。只回答"yes"或"no"。
法律条文：{document}
用户问题：{question}"""
        )
        self.grade_chain = grader_prompt | self.llm | StrOutputParser()

        legal_prompt = ChatPromptTemplate.from_template(
            """你是一个专业的中国法律顾问。请严格根据以下法律条文来回答用户的问题。

要求：
1. 只根据提供的法律条文回答，不要使用你自己的知识
2. 回答中请引用具体的法律名称和条款编号
3. 如果提供的条文中没有相关信息，请明确说"根据现有资料无法回答该问题"
4. 用通俗易懂的语言解释

参考法律条文：
{context}

用户问题：{question}"""
        )
        self.answer_chain = legal_prompt | self.llm | StrOutputParser()

        reject_prompt = ChatPromptTemplate.from_template(
            """用户问了一个与中国劳动法律无关的问题。请礼貌地告诉用户这个系统只能回答劳动法相关的问题，
并给出1-2个它能回答的问题示例。
用户问题：{question}"""
        )
        self.reject_chain = reject_prompt | self.llm | StrOutputParser()

    def _crag_filter(self, question: str, docs: List[Document]) -> List[Document]:
        relevant = []
        for i, doc in enumerate(docs):
            score = self.grade_chain.invoke({
                "document": doc.page_content,
                "question": question,
            }).strip().lower()
            tag = "✅" if score == "yes" else "❌"
            self._log(f"  {tag} Chunk {i}")
            if score == "yes":
                relevant.append(doc)
        if not relevant:
            self._log("  ⚠️ 全部不相关，保留得分最高的2个")
            return docs[:2]
        return relevant

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def query(self, question: str) -> Tuple[str, List[str]]:
        result = self.query_with_details(question)
        return result["answer"], result["contexts"]

    def query_with_details(self, question: str) -> Dict[str, Any]:
        self._log(f"\n{'=' * 60}")
        self._log(f"❓ 用户问题：{question}")
        self._log(f"{'=' * 60}")

        # Step 1: 路由
        route = self.route_chain.invoke({"question": question})
        self._log(f"\n🔀 Step 1 路由判断：{route}")

        if "知识库外" in route or "无关" in route:
            self._log("  → 非劳动法问题，礼貌拒答")
            answer = self.reject_chain.invoke({"question": question})
            return {
                "route": route, "answer": answer, "contexts": [],
                "chunks_retrieved": 0, "chunks_after_filter": 0,
            }

        # Step 2: Hybrid + Rerank
        if "咨询" in route:
            self._log("\n🔍 Step 2 检索策略：Hybrid + RAG-Fusion + Rerank")
            raw_docs = self._hybrid_rag_fusion_then_rerank(question)
        else:
            self._log("\n🔍 Step 2 检索策略：Hybrid + Rerank")
            raw_docs = self._hybrid_retrieve_then_rerank(question)
        self._log(f"  → 重排后保留 {len(raw_docs)} 个 chunk")

        # Step 3: CRAG
        self._log("\n🔎 Step 3 CRAG 质检：")
        filtered_docs = self._crag_filter(question, raw_docs)
        self._log(f"  → 质检后保留 {len(filtered_docs)} 个 chunk")

        # Step 4: 生成
        self._log("\n💬 Step 4 生成回答：")
        used_docs = filtered_docs[:4]
        context = "\n\n".join([doc.page_content for doc in used_docs])
        answer = self.answer_chain.invoke({"context": context, "question": question})
        self._log(answer)

        return {
            "route": route,
            "chunks_retrieved": len(raw_docs),
            "chunks_after_filter": len(filtered_docs),
            "answer": answer,
            "contexts": [doc.page_content for doc in used_docs],
        }


# ============================================================
# Demo
# ============================================================

def _run_demo():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    project_root = Path(__file__).resolve().parent.parent
    pipeline = HybridRerankLegalRAGPipeline(
        data_dir=str(project_root / "data"),
        persist_dir=str(project_root / "chroma_db"),
        verbose=True,
    )

    test_questions = [
        "试用期最长多久？",
        "保险事故发生后，被保险人应如何通知保险人？",
        "公司不给加班费怎么办？",
    ]

    for q in test_questions:
        pipeline.query_with_details(q)
        print()


if __name__ == "__main__":
    _run_demo()
