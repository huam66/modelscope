# modelscope
RAG AND QW（千问）
项目简介
本项目实现了一个轻量级、离线优先的 RAG（Retrieval-Augmented Generation）问答系统，专为中文技术文档（如《人工智能导论》《操作系统原理》等）设计。用户输入自然语言问题，系统自动从本地知识库中检索最相关段落并返回答案，适用于教学演示、内部知识库、技术面试准备等场景。
✨ 核心特性
✅ 中文优化：采用 BGE-small-zh-v1.5 嵌入模型，语义匹配准确
⚡ 极速响应：FAISS 向量索引 + GPU 加速，检索延迟 <100ms
💾 智能缓存：首次生成 embeddings 后自动缓存，重启秒加载
🌐 一键 Web UI：Gradio 构建交互界面，支持公网分享
📁 灵活接入：只需提供 chunks.json，即可切换任意中文知识库
🇨🇳 国内友好：自动使用 HF 镜像，魔搭/阿里云环境开箱即用
<img width="1182" height="636" alt="image" src="https://github.com/user-attachments/assets/370cdff9-b1d9-461d-a1f0-f2acb894312d" />

📦 技术栈
Embedding 模型: BAAI/bge-small-zh-v1.5
向量库: FAISS (FlatIP, L2 normalized)
Web 框架: Gradio
分块策略: RecursiveCharacterTextSplitter（保留完整句子）
运行环境: Python 3.9+, PyTorch, CUDA（可选）
人工智能通识教程pdf是RAG训练材料，21.pdf是微调材料
