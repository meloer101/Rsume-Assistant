import re
import os
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict


class RecursiveChunker:
    def __init__(self, chunk_size: int = 150, chunk_overlap_sentences: int = 1):
        # 中文简历每句约 30-60 字，150 字能装 2-4 句，是合理粒度
        self.chunk_size = chunk_size
        self.chunk_overlap_sentences = chunk_overlap_sentences
        self.separators = ["\n\n", "\n", "。", "；", " "]

    def _clean_pdf_text(self, text: str) -> str:
        # 1. 处理英文字母/数字之间的空格：E x c e l → Excel，3 0 % → 30%
        # 原理：字母或数字 + 单个空格 + 字母或数字，反复执行直到收敛
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r"([A-Za-z0-9])\s([A-Za-z0-9])", r"\1\2", text)

        # 2. 处理标准中文字符之间的空格（CJK unicode 范围）
        cjk = r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"
        prev = None
        while prev != text:
            prev = text
            text = re.sub(rf"({cjk})\s+({cjk})", r"\1\2", text)

        # 3. 处理 PDF 字体替换字符与普通字符之间的空格
        # ⽬⽤⼾⾏⼈⻔ 这类字符 unicode 在 \u2e80-\u2eff 和 \u2f00-\u2fdf 范围
        ext_cjk = r"[\u2e00-\u2eff\u2f00-\u2fdf\u3400-\u4dbf]"
        prev = None
        while prev != text:
            prev = text
            # 替换字符 ↔ 替换字符 之间的空格
            text = re.sub(rf"({ext_cjk})\s+({ext_cjk})", r"\1\2", text)
            # 替换字符 ↔ 标准中文 之间的空格
            text = re.sub(rf"({ext_cjk})\s+({cjk})", r"\1\2", text)
            text = re.sub(rf"({cjk})\s+({ext_cjk})", r"\1\2", text)
            # 替换字符 ↔ 英文 之间的空格
            text = re.sub(rf"({ext_cjk})\s+([A-Za-z0-9])", r"\1\2", text)
            text = re.sub(rf"([A-Za-z0-9])\s+({ext_cjk})", r"\1\2", text)

        # 4. 清理多余空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        中英文混合句子切割
        中文用：。；！？
        英文用：. ! ?
        """
        parts = re.split(r"(?<=[。；！？.!?])", text)
        return [s.strip() for s in parts if s.strip()]

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        parts = text.split(separator)
        return [p.strip() for p in parts if p.strip()]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        separator = separators[0]
        splits = self._split_by_separator(text, separator)

        result = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                result.extend(self._recursive_split(split, separators[1:]))
        return result

    def _merge_with_sentence_overlap(self, splits: list[str]) -> list[str]:
        chunks = []
        current_sentences = []
        current_len = 0

        for split in splits:
            sentences = self._split_into_sentences(split)
            for sentence in sentences:
                if current_len + len(sentence) + 1 <= self.chunk_size:
                    current_sentences.append(sentence)
                    current_len += len(sentence) + 1
                else:
                    if current_sentences:
                        chunks.append("".join(current_sentences))
                    overlap = current_sentences[-self.chunk_overlap_sentences :]
                    current_sentences = overlap + [sentence]
                    current_len = sum(len(s) + 1 for s in current_sentences)

        if current_sentences:
            chunks.append("".join(current_sentences))

        return chunks

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        if metadata is None:
            metadata = {}

        # 第一步先清洗，再切割
        cleaned_text = self._clean_pdf_text(text)

        raw_splits = self._recursive_split(cleaned_text, self.separators)
        merged = self._merge_with_sentence_overlap(raw_splits)

        return [
            Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(merged)},
            )
            for i, chunk_text in enumerate(merged)
        ]


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages]
        return "\n\n".join(filter(None, pages))
    except ImportError:
        raise ImportError("pip install pypdf")


def chunk_resume_pdf(pdf_path: str) -> list[Chunk]:
    text = extract_text_from_pdf(pdf_path)
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap_sentences=1)
    filename = os.path.basename(pdf_path)
    return chunker.chunk(text, metadata={"source": filename, "type": "resume"})


if __name__ == "__main__":
    chunks = chunk_resume_pdf("my_agent/文档/my_resume.pdf")

    print(f"共切出 {len(chunks)} 个 chunk\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} ({len(chunk.text)} 字符)")
        print(f"  内容: {chunk.text}")
        print(f"  元数据: {chunk.metadata}\n")
