import chromadb
from sentence_transformers import SentenceTransformer
from .chunker import chunk_resume_pdf  # 引入我们的 chunker

_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma_client.get_or_create_collection(
    name="pm_resume_knowledge",
    metadata={"hnsw:space": "cosine"}
)
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def ingest_resume_pdf(pdf_path: str) -> str:
    """
    把用户上传的简历 PDF 切块、embed、存入 ChromaDB。
    这个函数在用户上传文件时调用一次，不需要注册为 Agent Tool。
    
    Returns: 处理结果摘要，告知存了几个 chunk
    """
    chunks = chunk_resume_pdf(pdf_path)

    if not chunks:
        return "No content extracted from PDF."

    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    # ID 用文件名+chunk序号，避免重复上传时 ID 冲突
    import os
    filename = os.path.basename(pdf_path)
    ids = [f"resume_{filename}_chunk_{i}" for i in range(len(chunks))]

    embeddings = _embed_model.encode(texts).tolist()

    # upsert：同一份简历重复上传会覆盖，不会重复存储
    _collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return f"Ingested {len(chunks)} chunks from {filename} into knowledge base."


def retrieve_pm_knowledge(query: str) -> str:
    print(f"\n[RAG DEBUG] 收到的 query: {query}")  # 加这行
    
    query_embedding = _embed_model.encode(query).tolist()
    context_parts = []

    resume_results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
        where={"type": "resume"},
        include=["documents", "distances", "metadatas"]
    )
    for doc, dist in zip(resume_results["documents"][0], resume_results["distances"][0]):
        if 1 - dist > 0.05:
            context_parts.append(f"[user_resume]\n{doc}")

    for doc_type in ["before_after", "formula", "verb_rule"]:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            where={"type": doc_type},
            include=["documents", "distances"]
        )
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if 1 - dist > 0.1:
                context_parts.append(f"[{doc_type}]\n{doc}")

    result = "\n\n".join(context_parts) if context_parts else "No relevant knowledge found."
    
    print(f"[RAG DEBUG] 返回内容标签: {[p.split(chr(10))[0] for p in context_parts]}")  # 加这行
    
    return result