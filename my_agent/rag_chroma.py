import chromadb
from sentence_transformers import SentenceTransformer

# ChromaDB 本地持久化——数据存到磁盘，下次启动不需要重新 embed
# "./chroma_db" 是存储目录，会自动创建
client = chromadb.PersistentClient(path="./chroma_db")

# Collection 相当于一张"表"，专门存 PM 简历知识库
# get_or_create：存在就复用，不存在就新建
collection = client.get_or_create_collection(
    name="pm_resume_knowledge",
    metadata={"hnsw:space": "cosine"}  # 告诉 ChromaDB 用余弦相似度
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- 你的知识库文档 ----
# 注意结构：documents（原文）+ embeddings（向量）+ ids（唯一标识）+ metadatas（元数据）
# 元数据是关键——之后可以按 type 过滤检索
documents = [
    # Before/After 例子
    "BEFORE: Built a React dashboard. AFTER: Drove 40% reduction in reporting time by launching self-serve analytics dashboard, eliminating weekly manual reports for 3 teams",
    "BEFORE: Managed social media. AFTER: Owned growth strategy and content roadmap, growing Instagram following 3x in 4 months through weekly video series",
    "BEFORE: Helped with user research. AFTER: Spearheaded 12-user discovery sprint that surfaced 3 critical friction points, directly shaping Q3 feature prioritization",
    # PM 动词规则
    "Strong PM action verbs: Drove, Launched, Owned, Defined, Spearheaded, Prioritized, Shipped, Accelerated, Championed, Synthesized, Identified, Aligned",
    "Weak verbs to avoid in PM resumes: Built, Coded, Implemented, Developed, Created, Made, Helped, Worked on, Assisted, Supported",
    # Bullet 公式
    "PM bullet formula: [Strong verb] + [what you owned] + [measurable outcome]. Every bullet must answer: so what? who cared? how much?",
    "Metrics to use when you have no hard numbers: adoption rate, number of stakeholders aligned, time saved, features shipped, user interviews conducted, sprint velocity",
]

metadatas = [
    {"type": "before_after", "topic": "analytics"},
    {"type": "before_after", "topic": "growth"},
    {"type": "before_after", "topic": "research"},
    {"type": "verb_rule", "topic": "strong_verbs"},
    {"type": "verb_rule", "topic": "weak_verbs"},
    {"type": "formula", "topic": "bullet_structure"},
    {"type": "formula", "topic": "metrics"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

# Embed 并存入 ChromaDB
embeddings = model.encode(documents).tolist()  # ChromaDB 需要 list 格式

collection.upsert(  # upsert = 有就更新，没有就插入
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)

print(f"知识库已存储，共 {collection.count()} 条文档\n")

# ---- 检索测试 ----
def retrieve(query: str, n_results: int = 3, filter_type: str = None):
    query_embedding = model.encode(query).tolist()
    
    where = {"type": filter_type} if filter_type else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,  # 可以按 metadata 过滤，只检索特定类型
        include=["documents", "distances", "metadatas"]
    )
    return results

# 测试 1：通用检索
print("=== 测试 1：通用检索 ===")
results = retrieve("how to write a PM resume bullet about user research")
for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
    print(f"相似度: {1-dist:.3f} | type: {meta['type']}")
    print(f"  {doc[:80]}...\n")

# 测试 2：只检索 before_after 例子
print("=== 测试 2：只检索 before/after 例子 ===")
results = retrieve("rewrite this weak bullet", filter_type="before_after")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"相似度: {1-dist:.3f} | {doc[:80]}...")