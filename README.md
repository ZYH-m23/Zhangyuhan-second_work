# 2025秋 AI-NLP组第二次考核任务

**Topic：Transformer Architecture & Interpretability Circuits**

关于此作业的完整描述（包括架构细节、Gao et al. 2025 论文理论及具体任务），请参阅知识库内考核文档。

如果你在作业讲义或代码中发现任何问题，请随时提交 GitHub issue 或发起一个包含修复的 pull request。

---

## 配置与环境

我们使用 `uv` 来管理我们的环境，以确保可复现性、可移植性和易用性。

1.  请[在此处](https://github.com/astral-sh/uv)安装 `uv`（推荐），或者运行 `pip install uv`/`brew install uv`。
2.  我们建议你[在此处](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)阅读一些关于在 `uv` 中管理项目的信息。

现在，你可以使用以下命令来运行仓库中的任何代码（环境将在必要时被自动解析和激活）：

```sh
uv run <python_file_path>
```

---

## 代码实现与测试

本次考核仍然采用**测试驱动开发 (TDD)** 模式。你需要完成考核文档中要求的核心逻辑，并通过 `adapters.py` 将其连接到测试框架。

### 1. 运行单元测试

最初，所有的测试都应该因为 `NotImplementedError` 而失败。要将你的实现连接到测试，请在 [./tests/adapters.py](./tests/adapters.py) 中补完相应的函数接口。

运行所有测试：
```sh
uv run pytest
```

### 2. 针对特定组件的测试

为了方便调试，建议按模块顺序逐步通过测试。你可以使用 `-k` 参数运行特定的测试用例：

*   **基础组件 (Embedding, RMSNorm, Linear):**
    ```sh
    uv run pytest -k "test_embedding or test_rmsnorm or test_linear"
    ```

*   **前馈网络 (SwiGLU):**
    ```sh
    uv run pytest -k test_swiglu
    ```

*   **[可解释性] 稀疏激活 (AbsTopK):**
    ```sh
    uv run pytest -k test_abstopk
    ```

*   **注意力机制与位置编码 (RoPE, SDPA, Attention Sinks):**
    ```sh
    uv run pytest -k "test_rope or test_scaled_dot_product_attention or test_attention_with_sink"
    ```

*   **模型组装与剪枝分析:**
    ```sh
    uv run pytest -k "test_transformer_block or test_transformer_lm or test_magnitude_pruning"
    ```

---

## 下载数据

本次作业的微型回路发现（Narrator Circuit）任务需要使用 TinyStories 数据集。

```sh
mkdir -p data
cd data

# 下载 TinyStories 验证集样本用于调试和探针实验
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

---

## 致谢

感谢 [CS336](https://stanford-cs336.github.io/spring2025) 课程为本次作业提供的基础框架，以及 OpenAI 的论文 *Weight-sparse transformers have interpretable circuits (Gao et al., 2025)* 提供的理论支持。