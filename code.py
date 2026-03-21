#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 智能问答系统 - 基于《人工智能导论》PDF
功能：
1. 从 PDF 中提取文本并切分成知识块（若已有 chunks 文件则跳过）
2. 使用 BGE-small-zh-v1.5 模型生成嵌入向量，构建 FAISS 索引
3. 启动 Gradio Web 界面，支持自然语言问答
"""

import os
import re
import json
import numpy as np
from pathlib import Path

# ================================
# 1. 依赖安装提示
# ================================
try:
    import PyPDF2
    import faiss
    from sentence_transformers import SentenceTransformer
    import gradio as gr
    from modelscope import snapshot_download
except ImportError as e:
    print("缺少必要的依赖库，请先安装：")
    print("pip install PyPDF2 sentence-transformers faiss-cpu gradio modelscope torch numpy")
    raise e

# ================================
# 2. PDF 文本提取与分块
# ================================
def extract_and_split_ai_tutorial(pdf_path="人工智能通识教程.pdf", output_json="chunks_ai_tutorial.json"):
    """
    从 PDF 中提取正文，按标题切分成知识块，保存为 JSON 文件。
    """
    # 检查 PDF 文件是否存在
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"未找到文件: {pdf_file.absolute()}")

    # 读取整个 PDF
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    print(f"✅ 成功读取 PDF，总字符数: {len(full_text)}")

    # 定位正文起始位置（第一章）
    patterns = [
        r'第\s*1\s*章\s*[^\n]{0,20}人工智能',
        r'1\s*[\.．]\s*1\s*[^\n]{0,20}[Aa]rtificial',
        r'1\s*[\.．]\s*1\s*[^\n]{0,20}概述',
        r'第\s*一\s*章',
        r'1\s*[\.．]\s*1'
    ]
    main_start = 0
    for pattern in patterns:
        match = re.search(pattern, full_text)
        if match:
            main_start = match.start()
            print(f"🔍 找到正文起始位置 (匹配 '{match.group()}')")
            break
    else:
        print("⚠️ 未识别到第一章，将使用全文（可能包含前言/目录）")

    main_text = full_text[main_start:]

    # 按标题切分
    title_pattern = (
        r'('
        r'\d+[\.．]\d+(?:[\.．]\d+)*\s+[^。\n]{3,}'   # 1.1, 2.3.1
        r'|第\s*\d+\s*章\s+[^\n]{5,}'               # 第1章 XXX
        r'|第\s*[一二三四五六七八九十]+\s*章\s+[^\n]{5,}' # 第一章 XXX
        r')'
    )
    parts = re.split(title_pattern, main_text)
    chunks = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        full_chunk = f"{title}\n{content}".strip()
        if len(full_chunk) > 50:
            chunks.append(full_chunk)
    print(f"✂️ 初始切分得到 {len(chunks)} 个块")

    # 过滤非知识性章节
    skip_keywords = [
        "习题", "思考题", "练习", "参考文献", "附录", "索引",
        "目录", "前言", "序言", "内容简介", "版权", "防伪",
        "图书在版", "CIP", "主编", "副主编", "ISBN"
    ]
    filtered_chunks = []
    for chunk in chunks:
        first_line = chunk.split('\n')[0]
        if not any(kw in first_line for kw in skip_keywords):
            filtered_chunks.append(chunk)
    print(f"🧹 过滤后剩余 {len(filtered_chunks)} 个有效知识块")

    # 保存为 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存至: {Path(output_json).absolute()}")

    # 预览前3个块的标题
    print("\n📌 前3个块的标题预览:")
    for i, chunk in enumerate(filtered_chunks[:3]):
        title = chunk.split('\n')[0]
        print(f"  {i + 1}. {title}")

    return filtered_chunks


# ================================
# 3. 向量检索器类
# ================================
class AITutorialRetriever:
    def __init__(self, chunks_path="chunks_ai_tutorial.json", cache_dir="rag_cache"):
        self.chunks_path = chunks_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 加载文本块
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"❌ 找不到 chunks 文件: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        if not self.chunks:
            raise ValueError("❌ chunks 为空！请先提取并分块 PDF 文本。")
        print(f"📚 成功加载 {len(self.chunks)} 个文本块")

        # 初始化嵌入模型
        print("🧠 加载嵌入模型 BAAI/bge-small-zh-v1.5...")
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        print(f"✅ 模型加载完成，运行设备: {self.model.device}")

        # 向量维度
        self.embedding_dim = 512  # bge-small-zh-v1.5 输出维度固定

        # 尝试从缓存加载 FAISS 索引
        self.index_path = os.path.join(cache_dir, "faiss.index")
        self.embeddings_path = os.path.join(cache_dir, "embeddings.npy")

        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            print("📂 从缓存加载 FAISS 向量库...")
            self.index = faiss.read_index(self.index_path)
        else:
            print("⏳ 首次生成嵌入向量（稍等...）")
            embeddings = self.model.encode(
                self.chunks,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True
            ).astype('float32')

            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)

            # 保存缓存
            faiss.write_index(self.index, self.index_path)
            np.save(self.embeddings_path, embeddings)
            print(f"💾 向量库已缓存至 {cache_dir}/")

    def retrieve(self, query: str, k: int = 3):
        if not query or not query.strip():
            return []

        # 编码查询
        query_vec = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype('float32')

        # 检索
        distances, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= len(self.chunks):
                continue
            text = self.chunks[idx]
            title = (text.split('\n')[0] if '\n' in text else text[:30]).strip()
            score = float(distances[0][i])
            results.append({
                "title": title,
                "text": text,
                "score": max(0.0, score)
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)


# ================================
# 4. 问答函数
# ================================
def answer_question(query: str, retriever):
    if not query.strip():
        return "请输入问题。", ""

    results = retriever.retrieve(query, k=3)
    if not results:
        return "未找到相关资料。", ""

    main_answer = results[0]["text"]
    # 构建参考来源
    refs = "\n\n".join([
        f"【{i+1}】{res['title']} (相似度: {res['score']:.3f})\n{res['text'][:200]}..."
        for i, res in enumerate(results)
    ])
    return main_answer, refs


# ================================
# 5. 主入口
# ================================
if __name__ == "__main__":
    # 设置 Hugging Face 镜像（可选，用于国内加速）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 可选：魔搭模型下载（展示如何下载，但实际未使用）
    # model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
    # print(f"Qwen 模型下载至: {model_dir}")

    # 如果 chunks 文件不存在，则从 PDF 生成
    chunks_file = "chunks_ai_tutorial.json"
    if not os.path.exists(chunks_file):
        print("未找到 chunks 文件，开始从 PDF 提取...")
        extract_and_split_ai_tutorial(pdf_path="人工智能通识教程.pdf", output_json=chunks_file)
    else:
        print(f"使用已有 chunks 文件: {chunks_file}")

    # 初始化检索器
    retriever = AITutorialRetriever(chunks_path=chunks_file)

    # 构建 Gradio 界面
    with gr.Blocks(title="AI 教程问答助手") as demo:
        gr.Markdown("## 🤖 AI 教程智能问答系统")
        gr.Markdown("基于《人工智能导论》文档，使用 `bge-small-zh-v1.5` 向量检索")

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(label="请输入你的问题", placeholder="例如：人工智能的发展趋势是什么？", lines=2)
                submit_btn = gr.Button("🔍 搜索", variant="primary")
            with gr.Column(scale=3):
                answer = gr.Textbox(label="核心答案", interactive=False, lines=6)
                references = gr.Textbox(label="参考来源（Top 3）", interactive=False, lines=10)

        # 将 retriever 作为闭包传入
        submit_btn.click(
            fn=lambda q: answer_question(q, retriever),
            inputs=question,
            outputs=[answer, references]
        )

        gr.Examples(
            examples=[
                "人工智能的发展趋势是什么？",
                "哪些工作不容易被 AI 取代？",
                "AI 在医疗领域有哪些应用？"
            ],
            inputs=question
        )

    # 启动 Web 服务
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
