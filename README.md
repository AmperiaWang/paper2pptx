# Paper2PPT: Convert Your Research Papers to Powerpoint Slides Using LLMs

## 介绍

利用本地/在线 LLM，对传递上来的论文 pdf 作剖析，并生成精美的**图文** ppt：

- **文字**：由 LLM 按论文章节结构提炼（引言 / 相关工作 / 方法 / 实验 / 结论），自动平衡篇幅、拆分过长页面
- **图片**：从 PDF 原文中识别 `Figure N` 图注，导出为 `figureN.png`，并自动匹配到讨论该图的幻灯片；**无对应 Figure 的页面不插入任何图片**

默认使用项目内置模板 `template/index.pptx` 作为母版。

## 部署

```sh
git clone https://yet.not.upload/paper2ppt.git # 下载
pip install -r requirements.txt # 安装依赖
```

该工作默认的 backend 为 ollama（节省 token），若你没有大模型的 API，请在你的电脑上[安装 ollama](https://ollama.com/)：

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

并[挑选适配你的电脑的模型](https://ollama.com/search)，并部署。

## 执行

最简单的一行命令：

```sh
cd paper2ppt
python main.py input.pdf # input.pdf 换成你准备转成 PPT 的论文路径
```

其默认使用本地 ollama 作为解析论文所使用的引擎。若你有在线大模型的 API（`openai`、`deepseek`等），您可以指定：

```sh
cd paper2ppt
python main.py input.pdf --backend deepseek --apikey <你的 API key>
```

当使用默认的 ollama backend 时，其会自动搜索你电脑上面已经安装的最大模型（若无模型，其会发出提示）。若你希望指定模型（如 `qwen3:4b`），你可以通过 `--model` 参数指定：

```sh
cd paper2ppt
python main.py input.pdf --backend ollama --model qwen3:4b
```

在使用在线大模型的场景下，你也可以通过 `--model` 参数指定模型：

```sh
cd paper2ppt
python main.py input.pdf --backend openai --apikey <你的 API key> --model gpt-5.5
```

默认输出地址为与输入地址相同的同名 `.pptx` 文件；若你希望指定输出地址，请通过 `--output` 参数：

```sh
cd paper2ppt
python main.py input.pdf --output output.pptx
```

### 语言

通过 `--lang` 指定幻灯片文字语言，默认为 `zh_cn`（简体中文）：

```sh
python main.py input.pdf --lang zh_cn # 汉语
python main.py input.pdf --lang en_us # 美式英文
```

`zh_cn` 模式下，章节标题与正文均为简体中文（专有名词除外）。

### 模板

通过 `--template` 指定 PPT 模板，默认为 `template/index.pptx`：

```sh
python main.py input.pdf --template /path/to/your/template.pptx
```

程序会**删除模板中自带的示例页**，仅保留母版与版式，再生成新幻灯片。有 Figure 的页面会使用「图片与标题」版式；纯文字页不插入图片。

## 自定义

你可以任意修改 `prompt.json` 的内容（提示词）以使其适应不同的配置。
