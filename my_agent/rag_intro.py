from sentence_transformers import SentenceTransformer
import numpy as np

# 加载一个轻量级的 embedding 模型（首次运行会自动下载，约 90MB）
# 这个模型的作用：把一段文字 → 变成一个 384 维的数字向量
model = SentenceTransformer("all-MiniLM-L6-v2")

# 这就是你的"知识库"——先用 3 条硬编码数据理解原理
documents = [
    "Drove 40% reduction in onboarding time by launching a self-serve analytics dashboard",
    "Owned end-to-end delivery of a mobile feature adopted by 200k users in 3 months",
    "Built a React dashboard that shows user metrics and data visualizations",  # 故意放一条弱bullet
]

# 把每条文档变成向量
# embeddings 是一个 shape 为 (3, 384) 的 numpy 数组
embeddings = model.encode(documents)

print(f"每条文档变成了一个 {embeddings.shape[1]} 维的向量")
print(f"第一条文档的前 5 个数字: {embeddings[0][:5]}")

# ---- 核心概念：余弦相似度 ----
# 两个向量越"方向相似"，说明语义越接近
# 值域是 -1 到 1，越接近 1 越相似
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 用一个查询来检索最相关的文档
query = "impact-led PM resume bullet with metrics"
query_embedding = model.encode(query)

print(f"\n查询: '{query}'\n")

# 计算查询和每条文档的相似度
scores = []
for i, doc_embedding in enumerate(embeddings):
    score = cosine_similarity(query_embedding, doc_embedding)
    scores.append((score, documents[i]))
    print(f"相似度 {score:.3f} | {documents[i][:60]}...")

# 排序，取最相关的
scores.sort(reverse=True)
print(f"\n最相关的文档: {scores[0][1]}")
