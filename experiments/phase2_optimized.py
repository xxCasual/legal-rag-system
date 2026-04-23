import os
import shutil
os.environ["DEEPSEEK_API_KEY"] = "Your API Key"

# =====================================================================
# 基础设施：加载文档、切分、向量化（和Phase 1一样）
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
print(f"加载 {len(docs)} 个文档")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300,
    separators=["\n\n", "\n", "。", " ", ""]
)
splits = splitter.split_documents(docs)
print(f"切分为 {len(splits)} 个chunk")


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

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

##策略对比实验
# =====================================================================
# 策略1：Naive RAG（基线）
# =====================================================================
def naive_retrieve(question):
    return retriever.invoke(question)


# =====================================================================
# 策略2：RAG-Fusion（Multi-Query + RRF）
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
    return [(doc_map[c], s) for c, s in sorted_docs]


def rag_fusion_retrieve(question):
    queries = generate_queries.invoke({"question": question})
    print(f"  改写问题：")
    for i, q in enumerate(queries):
        print(f"    Q{i + 1}: {q}")

    all_queries = [question] + queries
    results_list = [retriever.invoke(q) for q in all_queries]
    fused = reciprocal_rank_fusion(results_list)
    return [doc for doc, score in fused]


# =====================================================================
# 策略3：HyDE（假回答检索）
# =====================================================================
hyde_prompt = ChatPromptTemplate.from_template(
    """请用1-2段话回答以下法律问题。不需要准确，只需要像法律教科书的语气写。
问题：{question}"""
)

generate_hypothesis = hyde_prompt | llm | StrOutputParser()

def hyde_retrieve(question):
    hypothesis = generate_hypothesis.invoke({"question": question})
    print(f"  假回答：{hypothesis[:100]}...")
    return retriever.invoke(hypothesis)


# =====================================================================
# 策略4：CRAG（检索后质检）
# =====================================================================
grader_prompt = ChatPromptTemplate.from_template(
    """判断以下法律条文是否与用户问题相关。只回答"yes"或"no"。

法律条文：{document}

用户问题：{question}"""
)

grade_chain = grader_prompt | llm | StrOutputParser()


def crag_retrieve(question):
    docs = retriever.invoke(question)
    relevant = []
    for i, doc in enumerate(docs):
        score = grade_chain.invoke({
            "document": doc.page_content,
            "question": question
        }).strip().lower()
        tag = "✅" if score == "yes" else "❌"
        print(f"  Chunk {i}: {tag}")
        if score == "yes":
            relevant.append(doc)

    if len(relevant) == 0:
        print(f"  ⚠️ 全部不相关，保留原始结果")
        return docs[:2]
    return relevant
# =====================================================================
# 对比实验
# =====================================================================
test_questions = [
    "试用期最长多久？",
    "公司不给加班费怎么办？",
    "什么情况下可以解除劳动合同？",
]
strategies = [
    ("Naive RAG", naive_retrieve),
    ("RAG-Fusion", rag_fusion_retrieve),
    ("HyDE", hyde_retrieve),
    ("CRAG", crag_retrieve),
]

for q in test_questions:
    print("\n" + "=" * 70)
    print(f"❓ 问题：{q}")
    print("=" * 70)

    for name, retrieve_fn in strategies:
        print(f"\n--- [{name}] ---")
        results = retrieve_fn(q)

        # 显示检索结果摘要
        print(f"  检索到 {len(results)} 个chunk：")
        for i, doc in enumerate(results[:4]):
            source = doc.metadata.get("source", "").split("/")[-1]
            # 检查是否包含关键法条
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"    #{i + 1} [{source}] {content_preview}...")

        # 生成回答
        context = "\n\n".join([doc.page_content for doc in results[:4]])
        answer = answer_chain.invoke({"context": context, "question": q})
        print(f"  💬 回答（前150字）：{answer[:150]}...")

    print()
