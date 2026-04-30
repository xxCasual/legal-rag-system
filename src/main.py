"""
src/main.py
============
法律 RAG 系统：路由 + RAG-Fusion + CRAG + 法律专用 Prompt

两种使用方式：
1. 直接跑 Demo（保留原行为）：
       python src/main.py

2. 作为模块被外部调用（用于 RAGAS 评估等）：
       from src.main import LegalRAGPipeline
       pipeline = LegalRAGPipeline(data_dir="data", verbose=False)
       answer, contexts = pipeline.query("未签合同有什么后果？")
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class LegalRAGPipeline:
    """
    法律 RAG 系统封装。一次实例化，重复调用 query()。
    """

    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        llm_model: str = "deepseek-chat",
        retriever_k: int = 4,
        rebuild_index: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            data_dir: 法律 txt 文件所在目录
            persist_dir: Chroma 持久化目录。已存在时复用，避免每次重建
            embedding_model: HuggingFace embedding 模型名
            llm_model: DeepSeek 模型名
            retriever_k: 单次检索返回 k 个 chunk
            rebuild_index: True 时强制重建索引（修改了文档或 chunk 策略时用）
            verbose: 是否打印详细过程。评估时设 False 避免刷屏
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.retriever_k = retriever_k
        self.verbose = verbose

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "请设置 DEEPSEEK_API_KEY 环境变量。\n"
                "  方式一: export DEEPSEEK_API_KEY=sk-xxx\n"
                "  方式二: 项目根目录创建 .env 文件，内容: DEEPSEEK_API_KEY=sk-xxx"
            )

        # ---- 准备向量库 ----
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        persist_path = Path(persist_dir)
        index_exists = persist_path.exists() and any(persist_path.iterdir())

        if rebuild_index or not index_exists:
            if persist_path.exists():
                shutil.rmtree(persist_dir)
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
            splits = splitter.split_documents(docs)
            self.vectorstore = Chroma.from_documents(
                splits, embeddings, persist_directory=persist_dir
            )
            self._log(f"✅ 索引已构建: {len(docs)} 文档, {len(splits)} chunks")
        else:
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
            )
            self._log(f"✅ 索引已加载（复用）: {persist_dir}")

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": retriever_k}
        )

        # ---- LLM ----
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        # ---- 各个 chain ----
        self._build_chains()

    # ----------------------------------------------------------------
    # 内部：日志（受 verbose 控制）
    # ----------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ----------------------------------------------------------------
    # 内部：构建 prompts 和 chains
    # ----------------------------------------------------------------

    def _build_chains(self):
        # 路由器
        router_prompt = ChatPromptTemplate.from_template(
            """你是一个法律问题分类器。本系统的知识库包含 4 部中国法律：
        《劳动法》、《劳动合同法》、《劳动争议调解仲裁法》、《保险法》。

        请判断用户问题属于以下哪个类型，只回答类型名称：

        - "法条查询"：询问具体的法律条文规定（如"试用期最长多久""保险金额怎么定"）
        - "法律咨询"：描述具体情境寻求建议（如"公司不给加班费怎么办""保险公司拒赔怎么办"）  
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

        # CRAG 质检
        grader_prompt = ChatPromptTemplate.from_template(
            """判断以下法律条文是否与用户问题相关。只回答"yes"或"no"。
法律条文：{document}
用户问题：{question}"""
        )
        self.grade_chain = grader_prompt | self.llm | StrOutputParser()

        # 法律生成
        legal_prompt = ChatPromptTemplate.from_template(
            """你是一个专业的中国劳动法律顾问。请严格根据以下法律条文来回答用户的问题。

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
    # 内部：RAG-Fusion + CRAG
    # ----------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(results_list, k: int = 60):
        scores, doc_map = {}, {}
        for result in results_list:
            for rank, doc in enumerate(result, start=1):
                content = doc.page_content
                if content not in scores:
                    scores[content] = 0
                    doc_map[content] = doc
                scores[content] += 1 / (k + rank)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[c] for c, _ in sorted_docs]

    def _rag_fusion_retrieve(self, question: str):
        queries = self.generate_queries.invoke({"question": question})
        self._log("  📝 RAG-Fusion改写：")
        for i, q in enumerate(queries):
            self._log(f"     Q{i + 1}: {q}")
        all_queries = [question] + queries
        results_list = [self.retriever.invoke(q) for q in all_queries]
        return self._reciprocal_rank_fusion(results_list)

    def _crag_filter(self, question: str, docs):
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
        """
        外部接口（RAGAS 评估用）。
        Returns: (answer, contexts) — contexts 是真正喂给生成器的 chunk 文本
        """
        result = self.query_with_details(question)
        return result["answer"], result["contexts"]

    def query_with_details(self, question: str) -> Dict[str, Any]:
        """完整查询，返回所有中间产物（demo / 调试用）。"""
        self._log(f"\n{'=' * 60}")
        self._log(f"❓ 用户问题：{question}")
        self._log(f"{'=' * 60}")

        # Step 1: 路由
        route = self.route_chain.invoke({"question": question})
        self._log(f"\n🔀 Step 1 路由判断：{route}")

        # 知识库外问题：礼貌拒答
        if "知识库外" in route or "无关" in route:
            self._log("  → 非劳动法问题，礼貌拒答")
            answer = self.reject_chain.invoke({"question": question})
            self._log(f"\n💬 回答：\n{answer}")
            return {
                "route": route,
                "answer": answer,
                "contexts": [],
                "chunks_retrieved": 0,
                "chunks_after_filter": 0,
            }

        # Step 2: 检索
        if "咨询" in route:
            self._log("\n🔍 Step 2 检索策略：RAG-Fusion（口语化问题）")
            raw_docs = self._rag_fusion_retrieve(question)
        else:
            self._log("\n🔍 Step 2 检索策略：直接检索（精确法条查询）")
            raw_docs = self.retriever.invoke(question)
        self._log(f"  → 检索到 {len(raw_docs)} 个chunk")

        # Step 3: CRAG 质检
        self._log("\n🔎 Step 3 CRAG质检：")
        filtered_docs = self._crag_filter(question, raw_docs[:6])
        self._log(f"  → 质检后保留 {len(filtered_docs)} 个chunk")

        # Step 4: 生成回答
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
    """保留原始 demo 行为：跑 6 个测试问题 + 汇总报告"""
    # 自动加载项目根目录的 .env（如果有）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 数据目录：相对当前文件向上一级 / data
    data_dir = Path(__file__).resolve().parent.parent / "data"

    pipeline = LegalRAGPipeline(data_dir=str(data_dir), verbose=True)

    test_questions = [
        # 法条查询类
        "试用期最长多久？",
        "经济补偿金怎么计算？",
        # 法律咨询类
        "公司不给加班费怎么办？",
        "老板让我签空白合同，我该怎么办？",
        # 知识库外
        "今天天气怎么样？",
        "Python怎么写for循环？",
    ]

    results = []
    for q in test_questions:
        result = pipeline.query_with_details(q)
        results.append({"question": q, **result})
        print()

    # 汇总
    print("\n" + "=" * 60)
    print("📊 系统测试汇总报告")
    print("=" * 60)
    for r in results:
        route = r.get("route", "")
        chunks = r.get("chunks_after_filter", 0)
        print(f"  [{route}] {r['question']}")
        print(f"    → 使用 {chunks} 个chunk生成回答")
        print()


if __name__ == "__main__":
    _run_demo()
