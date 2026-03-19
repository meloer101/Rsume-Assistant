import sys
from .rag_tool import ingest_resume_pdf

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "my_agent/文档/my_resume.pdf"
    result = ingest_resume_pdf(pdf_path)
    print(result)
