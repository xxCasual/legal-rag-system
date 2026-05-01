"""
src/hybrid_main.py
==================
Hybrid 检索版法律 RAG。

与 src/main.py 唯一区别：把单一向量检索替换为 BM25+向量+RRF 融合。
其他模块（路由器 / RAG-Fusion 多查询 / CRAG / 法律 Prompt）完全一致。

设计动机：
- baseline 评估发现 30% 样本"漏召回"关键法条
- BM25 对条号、专业术语精确匹配能力强，能补充向量检索盲区
- RRF（Reciprocal Rank Fusion）将两种检索结果按排名融合，无需归一化分数
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


class HybridLegalRAGPipeline:
    """
    Hybrid 检索版法律 RAG。
    """

    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        llm_model: str = "deepseek-chat",
        retriever_k: int = 4,
        bm25_k: int = 4,
        vector_k: int = 4,
        rrf_k_const: int = 60,
        rebuild_index: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            retriever_k: 最终融合后返回的 chunk 数
            bm25_k: BM25 检索召回数（建议 ≥ retriever_k）
            vector_k: 向量检索召回数（建议 ≥ retriever_k）
            rrf_k_const: RRF 公式中的 k 常数，60 是论文推荐值
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.retriever_k = retriever_k
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.rrf_k_const = rrf_k_const
        self.verbose = verbose

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "请设置 DEEPSEEK_API_KEY 环境变量。"
                " export DEEPSEEK_API_KEY=sk-xxx 或在 .env 中配置。"
            )

        # ---- 加载文档（BM25 必须保留全部 chunks 在内存）----
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

        # ---- 向量检索 ----
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

        # ---- BM25 检索 ----
        # BM25Retriever.from_documents 内部会做分词。中文场景需要手工分词后传入
        # 这里用一个简单的字符级 + 数字/英文保留的分词，对法律文本足够
        self.bm25_retriever = self._build_bm25_retriever(self.splits, k=bm25_k)
        self._log(f"✅ BM25 索引已构建")

        # ---- LLM ----
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        # ---- 各 chain ----
        self._build_chains()

    # ----------------------------------------------------------------
    # 工具：日志
    # ----------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ----------------------------------------------------------------
    # BM25：中文分词
    # ----------------------------------------------------------------

    @staticmethod
    def _chinese_tokenize(text: str) -> List[str]:
        """
        极简中文分词：单字 + 连续数字/英文/法条编号保留。
        对法律文本足够：
        - "第二十一条" 会被切成 ['第','二','十','一','条']，BM25 仍能命中
        - 想要更好的效果可以换 jieba：
              import jieba
              return list(jieba.cut(text))
        """
        import re
        # 把连续的数字/英文/特殊符号识别出来，其余按单字
        tokens = []
        # 匹配：数字串 / 英文串 / 单个汉字 / 单个其他字符
        pattern = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]|[^\s]")
        for m in pattern.finditer(text):
            tokens.append(m.group())
        return tokens

    def _build_bm25_retriever(
        self, documents: List[Document], k: int
    ) -> BM25Retriever:
        retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=self._chinese_tokenize,
        )
        retriever.k = k
        return retriever

    # ----------------------------------------------------------------
    # 检索：BM25 + 向量 + RRF 融合
    # ----------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        results_lists: List[List[Document]], k_const: int = 60
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion。
        每个文档的得分 = Σ 1/(k_const + rank_i)，rank 从 1 开始。
        """
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

    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """单次 hybrid 检索：BM25 + 向量 + RRF 融合"""
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        fused = self._rrf_fuse([bm25_docs, vector_docs], k_const=self.rrf_k_const)
        return fused[: self.retriever_k]

    def _hybrid_rag_fusion_retrieve(self, question: str) -> List[Document]:
        """
        多查询 + Hybrid + RRF：
        1. 用 LLM 把原 query 改写成 3 个查询
        2. 每个查询都做 BM25+向量 hybrid 检索
        3. 把所有 hybrid 结果再用一次 RRF 融合
        """
        queries = self.generate_queries.invoke({"question": question})
        self._log("  📝 RAG-Fusion 改写：")
        for i, q in enumerate(queries):
            self._log(f"     Q{i + 1}: {q}")

        all_queries = [question] + queries
        # 每个 query 做一次 hybrid
        per_query_results = [self._hybrid_retrieve(q) for q in all_queries]
        # 再 RRF 一次
        fused = self._rrf_fuse(per_query_results, k_const=self.rrf_k_const)
        return fused

    # ----------------------------------------------------------------
    # Chains（与 main.py 一致）
    # ----------------------------------------------------------------

    def _build_chains(self):
        # 路由器（用 main.py 改进过的版本）
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

        # 多查询改写
        multi_query_prompt = ChatPromptTemplate.from_template(
            """你是一个法律检索助手。请把用户的法律问题改写成3个不同角度的版本，
用于在法律法规数据库中检索。每个版本一行，不要编号，不要额外说明。
原始问题：{question}"""
        )
        self.generate_queries = (
            multi_query_prompt | self.llm | StrOutputParser()
            | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()])
        )

        # CRAG
        grader_prompt = ChatPromptTemplate.from_template(
            """判断以下法律条文是否与用户问题相关。只回答"yes"或"no"。
法律条文：{document}
用户问题：{question}"""
        )
        self.grade_chain = grader_prompt | self.llm | StrOutputParser()

        # 法律生成
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

        # 拒答
        reject_prompt = ChatPromptTemplate.from_template(
            """用户问了一个与中国劳动法律无关的问题。请礼貌地告诉用户这个系统只能回答劳动法相关的问题，
并给出1-2个它能回答的问题示例。
用户问题：{question}"""
        )
        self.reject_chain = reject_prompt | self.llm | StrOutputParser()

    # ----------------------------------------------------------------
    # CRAG
    # ----------------------------------------------------------------

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
        """RAGAS 评估接口。"""
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

        # Step 2: Hybrid 检索
        if "咨询" in route:
            self._log("\n🔍 Step 2 检索策略：Hybrid + RAG-Fusion")
            raw_docs = self._hybrid_rag_fusion_retrieve(question)
        else:
            self._log("\n🔍 Step 2 检索策略：Hybrid（BM25+向量）")
            raw_docs = self._hybrid_retrieve(question)
        self._log(f"  → 检索到 {len(raw_docs)} 个 chunk")

        # Step 3: CRAG
        self._log("\n🔎 Step 3 CRAG 质检：")
        filtered_docs = self._crag_filter(question, raw_docs[:6])
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
    pipeline = HybridLegalRAGPipeline(
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
