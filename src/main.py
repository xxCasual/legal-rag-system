import os
import shutil

os.environ["DEEPSEEK_API_KEY"] = "Your API Key"

# =====================================================================
# 基础设施
# =====================================================================
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

loader = DirectoryLoader("../data/", glob="*.txt", loader_cls=TextLoader,
                         loader_kwargs={"encoding": "utf-8"})
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300,
    separators=["\n\n", "\n", "。", " ", ""]
)
splits = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

print(f"✅ 系统初始化完成：{len(docs)}个文档，{len(splits)}个chunk")

# =====================================================================
# 模块1：Query路由器
# 判断问题属于哪种类型，决定走哪条检索路径
# =====================================================================
router_prompt = ChatPromptTemplate.from_template(
    """你是一个法律问题分类器。请判断用户的问题属于以下哪个类型，只回答类型名称：

- "法条查询"：用户在问某个具体的法律条文规定，比如"试用期最长多久"、"加班费标准是什么"
- "法律咨询"：用户在描述一个具体情况寻求建议，比如"公司不给加班费怎么办"、"被辞退了能要赔偿吗"
- "知识库外"：问题和中国劳动法律完全无关，比如天气、体育、技术问题

用户问题：{question}
类型："""
)

route_chain = router_prompt | llm | StrOutputParser() | (lambda x: x.strip())

# =====================================================================
# 模块2：RAG-Fusion检索（用于法律咨询类问题）
# =====================================================================
multi_query_prompt = ChatPromptTemplate.from_template(
    """你是一个法律检索助手。请把用户的法律问题改写成3个不同角度的版本，
用于在法律法规数据库中检索。每个版本一行，不要编号，不要额外说明。
原始问题：{question}"""
)

generate_queries = (
        multi_query_prompt | llm | StrOutputParser()
        | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip()])
)


def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    doc_map = {}
    for result in results_list:
        for rank, doc in enumerate(result, start=1):
            content = doc.page_content
            if content not in scores:
                scores[content] = 0
                doc_map[content] = doc
            scores[content] += 1 / (k + rank)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[c] for c, s in sorted_docs]


def rag_fusion_retrieve(question):
    queries = generate_queries.invoke({"question": question})
    print(f"  📝 RAG-Fusion改写：")
    for i, q in enumerate(queries):
        print(f"     Q{i + 1}: {q}")
    all_queries = [question] + queries
    results_list = [retriever.invoke(q) for q in all_queries]
    return reciprocal_rank_fusion(results_list)


# =====================================================================
# 模块3：CRAG质检
# =====================================================================
grader_prompt = ChatPromptTemplate.from_template(
    """判断以下法律条文是否与用户问题相关。只回答"yes"或"no"。
法律条文：{document}
用户问题：{question}"""
)

grade_chain = grader_prompt | llm | StrOutputParser()


def crag_filter(question, docs):
    relevant = []
    for i, doc in enumerate(docs):
        score = grade_chain.invoke({
            "document": doc.page_content,
            "question": question
        }).strip().lower()
        tag = "✅" if score == "yes" else "❌"
        print(f"  {tag} Chunk {i}")
        if score == "yes":
            relevant.append(doc)

    if len(relevant) == 0:
        print(f"  ⚠️ 全部不相关，保留得分最高的2个")
        return docs[:2]
    return relevant


# =====================================================================
# 模块4：法律专用生成
# =====================================================================
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

answer_chain = legal_prompt | llm | StrOutputParser()

reject_prompt = ChatPromptTemplate.from_template(
    """用户问了一个与中国劳动法律无关的问题。请礼貌地告诉用户这个系统只能回答劳动法相关的问题，
并给出1-2个它能回答的问题示例。
用户问题：{question}"""
)

reject_chain = reject_prompt | llm | StrOutputParser()


# =====================================================================
# 主流程：组装完整系统
# =====================================================================
def legal_rag_system(question: str):
    """完整的法律RAG系统：路由 → 检索 → 质检 → 生成"""

    print(f"\n{'=' * 60}")
    print(f"❓ 用户问题：{question}")
    print(f"{'=' * 60}")

    # Step 1: 路由
    route = route_chain.invoke({"question": question})
    print(f"\n🔀 Step 1 路由判断：{route}")

    # Step 1分支：知识库外问题直接拒答
    if "知识库外" in route or "无关" in route:
        print(f"  → 非劳动法问题，礼貌拒答")
        answer = reject_chain.invoke({"question": question})
        print(f"\n💬 回答：\n{answer}")
        return {"route": route, "answer": answer, "chunks_used": 0}

    # Step 2: 根据路由类型选择检索策略
    if "咨询" in route:
        print(f"\n🔍 Step 2 检索策略：RAG-Fusion（口语化问题，需要多角度检索）")
        raw_docs = rag_fusion_retrieve(question)
    else:
        print(f"\n🔍 Step 2 检索策略：直接检索（精确法条查询）")
        raw_docs = retriever.invoke(question)

    print(f"  → 检索到 {len(raw_docs)} 个chunk")

    # Step 3: CRAG质检
    print(f"\n🔎 Step 3 CRAG质检：")
    filtered_docs = crag_filter(question, raw_docs[:6])
    print(f"  → 质检后保留 {len(filtered_docs)} 个chunk")

    # Step 4: 生成回答
    print(f"\n💬 Step 4 生成回答：")
    context = "\n\n".join([doc.page_content for doc in filtered_docs[:4]])
    answer = answer_chain.invoke({"context": context, "question": question})
    print(answer)

    return {
        "route": route,
        "chunks_retrieved": len(raw_docs),
        "chunks_after_filter": len(filtered_docs),
        "answer": answer
    }


# =====================================================================
# 测试：覆盖三种路由类型
# =====================================================================
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
    result = legal_rag_system(q)
    results.append({"question": q, **result})
    print()

# =====================================================================
# 汇总报告
# =====================================================================
print("\n" + "=" * 60)
print("📊 系统测试汇总报告")
print("=" * 60)
for r in results:
    route = r.get("route", "")
    chunks = r.get("chunks_after_filter", r.get("chunks_used", 0))
    print(f"  [{route}] {r['question']}")
    print(f"    → 使用 {chunks} 个chunk生成回答")
    print()
