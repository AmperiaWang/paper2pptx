"""
paper2ppt 包根模块。

本包实现论文 PDF → LLM 分析 → PowerPoint 的完整转换流程。
子模块职责：
  - pdf_parser   : 从 PDF 提取文本与图片
  - prompts      : 加载 prompt.json、解析 LLM 返回的 JSON
  - llm          : 各 LLM 后端的 HTTP 客户端（ollama / openai / deepseek）
  - ppt_generator: 将结构化幻灯片数据写入 .pptx
  - pipeline     : 串联上述步骤的主流程
"""
