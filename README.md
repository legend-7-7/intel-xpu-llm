# 基于 PyTorch XPU 的本地知识库问答助手

## 1. 项目简介

本项目是一个基于本地大语言模型和 RAG 技术的个人知识库问答系统。

项目面向本地设备运行，重点探索在 Intel Core Ultra 处理器和 Intel Arc 核显环境下，使用 PyTorch XPU 进行本地 AI 应用开发的流程。

系统目标是实现以下功能：

- 检测本地 CPU / Intel XPU / PyTorch 环境
- 读取本地文档
- 将长文本切分为文本块
- 使用 embedding 模型生成文本向量
- 构建本地向量知识库
- 根据用户问题检索相关文档片段
- 调用本地语言模型生成回答
- 使用 Streamlit 构建可视化交互界面

## 2. 项目背景

随着大语言模型的发展，越来越多的 AI 应用开始从云端服务转向本地部署。本地运行大模型具有数据隐私更好、可控性更强、离线可用等优势。

同时，Intel Core Ultra 系列处理器集成了 CPU、GPU 和 NPU 等异构计算单元。其中 Intel Arc 核显可以通过 PyTorch XPU 后端参与 AI 推理任务。本项目基于这一硬件环境，尝试构建一个能够在本地设备上运行的知识库问答系统。

本项目不是训练大模型，而是重点实现本地推理和 RAG 应用流程。

## 3. 技术路线

项目整体流程如下：

```text
设备环境检测
→ 本地文档读取
→ 文本切分
→ 文本向量化
→ 向量数据库构建
→ 问题检索
→ RAG 上下文构造
→ 本地 LLM 生成回答
→ Streamlit 可视化展示
````

## 4. 当前设备环境

当前开发设备检测结果：

```text
Python version: 3.11.15
PyTorch version: 2.12.0+xpu
XPU available: True
XPU count: 1
XPU device name: Intel(R) Arc(TM) Graphics
Selected device: xpu
CPU: Intel(R) Core(TM) Ultra 7 258V
Memory: 30 GiB
```

说明当前设备已经能够被 PyTorch XPU 正常识别。

## 5. 项目结构

```text
intel-xpu-llm/
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── device_manager.py
│   ├── llm_client.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embedding_model.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── utils.py
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Device_Check.py
│       ├── 2_Document_Manager.py
│       ├── 3_Knowledge_Index.py
│       ├── 4_RAG_Chat.py
│       └── 5_System_Info.py
├── data/
│   ├── documents/
│   ├── chunks/
│   └── vector_db/
├── models/
├── results/
├── docs/
│   └── report.md
└── tests/
```

## 6. 功能模块说明

### 6.1 设备检测模块

对应文件：

```text
src/device_manager.py
```

功能：

* 检测 Python 版本
* 检测 PyTorch 版本
* 检测 Intel XPU 是否可用
* 获取 CPU 和内存信息
* 自动选择运行设备

### 6.2 本地 LLM 调用模块

对应文件：

```text
src/llm_client.py
```

功能：

* 加载本地语言模型
* 管理模型推理设备
* 根据 prompt 生成回答

### 6.3 文档读取模块

对应文件：

```text
src/document_loader.py
```

功能：

* 读取本地 txt、md、py 等文本文件
* 提取文件内容和元数据
* 为后续知识库构建提供原始文本

### 6.4 文本切分模块

对应文件：

```text
src/text_splitter.py
```

功能：

* 将长文档切分为较短文本块
* 支持重叠切分
* 保存文本块和来源信息

### 6.5 Embedding 模块

对应文件：

```text
src/embedding_model.py
```

功能：

* 加载 embedding 模型
* 将文本块转换为向量
* 为语义检索提供向量表示

### 6.6 向量数据库模块

对应文件：

```text
src/vector_store.py
```

功能：

* 保存文本向量
* 管理本地向量数据库
* 支持相似度检索

### 6.7 检索模块

对应文件：

```text
src/retriever.py
```

功能：

* 根据用户问题检索相关文本块
* 返回相似度最高的文档片段
* 提供 RAG 上下文来源

### 6.8 RAG 问答模块

对应文件：

```text
src/rag_pipeline.py
```

功能：

* 接收用户问题
* 检索相关文档内容
* 构造提示词
* 调用本地 LLM 生成回答
* 返回答案和引用来源

### 6.9 Streamlit 可视化模块

对应目录：

```text
app/
```

功能：

* 设备检测页面
* 文档管理页面
* 知识库索引页面
* RAG 对话页面
* 系统信息页面

## 7. 环境配置

本项目推荐使用 Conda 创建 Python 3.11 环境：

```bash
conda create -n intel-xpu-llm python=3.11 -y
conda activate intel-xpu-llm
```

安装 PyTorch XPU 版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

检查 XPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.device_count() if torch.xpu.is_available() else 0)"
```

## 8. 运行方式

### 8.1 设备检测

```bash
python src/device_manager.py
```

或者：

```bash
python main.py
```

### 8.2 启动 Streamlit 系统

后续完成可视化系统后，可使用：

```bash
streamlit run app/Home.py
```

## 9. Git 分支说明

本项目采用标准 GitHub 分支管理流程：

```text
main：最终稳定版本
dev：开发整合分支
feature/*：功能开发分支
```

计划功能分支包括：

```text
feature/project-structure
feature/device-check
feature/readme
feature/local-llm-setup
feature/document-loader
feature/text-splitting
feature/embedding-vector-store
feature/rag-pipeline
feature/streamlit-ui
feature/report
```

开发流程：

```text
从 dev 创建 feature 分支
→ 完成单个功能
→ commit
→ push 到 GitHub
→ 创建 Pull Request
→ 合并到 dev
→ 删除 feature 分支
→ 最后 dev 合并到 main
```

## 10. 当前进度

* [x] 项目结构搭建
* [x] PyTorch XPU 环境检测
* [x] README 初稿
* [ ] 本地 LLM 调用
* [ ] 文档读取
* [ ] 文本切分
* [ ] Embedding 向量化
* [ ] 向量数据库
* [ ] RAG 问答流程
* [ ] Streamlit 可视化系统
* [ ] 项目报告

## 11. 项目说明

本项目主要用于学习和实践本地 AI 应用开发流程。项目重点不在于训练大模型，而在于基于本地硬件、PyTorch XPU 和 RAG 技术构建一个完整的个人知识库问答系统。

由于本地模型推理受到设备性能、模型大小和内存限制影响，第一版将优先保证系统流程完整和稳定运行，后续再逐步优化模型效果和推理速度。
