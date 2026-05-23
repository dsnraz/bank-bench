"""
MemoryBank-SiliconFriend 记忆摘要流水线（handler 模式）。

对标 hyperbolic_memory 的 session_run.py 架构:
  - 抽取模型: 生成逐日对话摘要 + 人格分析
  - 生成模型: 生成整体历史汇总 + 整体人格汇总
  - 嵌入模型: all-mpnet-base-v2

核心逻辑全部调用官方模块 memory_bank.summarize_memory。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 确保项目根目录和 memory_bank 在 sys.path 中
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "memory_bank"))

from model_handler import create_model_handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MemoryBank-SiliconFriend 记忆摘要流水线"
    )

    # 数据路径
    p.add_argument("--memory-dir", type=str, required=True,
                   help="记忆库 JSON 文件路径（如 memories/eng_memory_cases.json）")
    p.add_argument("--out-file", type=str, default=None,
                   help="输出 JSON 路径（默认覆盖原文件）")
    p.add_argument("--user-name", type=str, default=None,
                   help="指定用户名，不传则处理所有用户")
    p.add_argument("--language", type=str, default="en", choices=("cn", "en"))

    # ── 抽取模型（Phase 1: 逐日摘要 + 人格分析）───────────────────
    p.add_argument("--extraction-handler-type", type=str, default="transformers",
                   choices=("transformers", "openai"),
                   help="抽取模型后端: transformers=本地llama/qwen, openai=DeepSeek API")
    p.add_argument("--extraction-model-path", type=str, default=None,
                   help="抽取模型路径 (transformers) 或模型名 (openai, 默认 deepseek-chat)")
    p.add_argument("--extraction-api-base", type=str, default="https://api.deepseek.com")
    p.add_argument("--extraction-api-key", type=str, default=None)

    # ── 生成模型（Phase 2: 整体历史 + 人格汇总）────────────────────
    p.add_argument("--generation-handler-type", type=str, default="transformers",
                   choices=("transformers", "openai"),
                   help="生成模型后端: transformers=本地llama/qwen, openai=DeepSeek API")
    p.add_argument("--generation-model-path", type=str, default=None,
                   help="生成模型路径 (transformers) 或模型名 (openai, 默认 deepseek-chat)")
    p.add_argument("--generation-api-base", type=str, default="https://api.deepseek.com")
    p.add_argument("--generation-api-key", type=str, default=None)

    # ── 嵌入模型 ─────────────────────────────────────────────────
    p.add_argument("--embedding-model", type=str,
                   default="sentence-transformers/all-mpnet-base-v2",
                   help="句向量模型名或本地路径")

    # ── 模式 ─────────────────────────────────────────────────────
    p.add_argument("--locomo", action="store_true", default=False,
                   help="LoCoMo 模式：嵌套结构 sample→session，name[0]/name[1] 作为对话双方")

    # ── 运行控制 ─────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _load_handler(handler_type: str, model_path: str | None,
                  api_base: str, api_key: str | None,
                  device: str, label: str = ""):
    """加载 handler（transformers 或 openai）。"""
    if handler_type == "transformers":
        if not model_path:
            print(f"[{label}] transformers 类型必须提供 model-path，跳过（回退到原 OpenAI 逻辑）")
            return None
        handler = create_model_handler("transformers")
        ok = handler.load(model_path, device=device)
        if not ok:
            raise RuntimeError(f"[{label}] 模型加载失败: {model_path}")
        return handler
    elif handler_type == "openai":
        model_name = model_path or "deepseek-chat"
        handler = create_model_handler("openai", api_base=api_base)
        ok = handler.load(model_name, device=device, api_key=api_key)
        if not ok:
            raise RuntimeError(f"[{label}] API 连接失败: {api_base}")
        return handler
    else:
        raise ValueError(f"未知 handler 类型: {handler_type}")


def main() -> None:
    args = parse_args()

    # 1. 加载抽取 handler
    extraction_handler = _load_handler(
        args.extraction_handler_type, args.extraction_model_path,
        args.extraction_api_base, args.extraction_api_key,
        args.device, label="抽取模型"
    )

    # 2. 加载生成 handler（如果和抽取不同）
    generation_handler = None
    if args.generation_handler_type != args.extraction_handler_type or \
       args.generation_model_path != args.extraction_model_path:
        generation_handler = _load_handler(
            args.generation_handler_type, args.generation_model_path,
            args.generation_api_base, args.generation_api_key,
            args.device, label="生成模型"
        )

    # 3. 设置嵌入模型（覆盖 model_config 默认值）
    from memory_bank.memory_retrieval.configs import model_config
    model_config.EMBEDDING_MODEL_EN = args.embedding_model
    print(f"[嵌入模型] {args.embedding_model}")

    # 4. 调用官方模块 summarize_memory — 核心逻辑不动
    from memory_bank.summarize_memory import summarize_memory

    memory_dir = args.memory_dir
    out_file = args.out_file or memory_dir
    print(f"输入记忆库: {memory_dir}")
    print(f"输出文件:   {out_file}")

    # 如果输出路径不同于输入，先复制一份
    if out_file != memory_dir:
        import shutil
        shutil.copy(memory_dir, out_file)

    result = summarize_memory(
        out_file,
        name=args.user_name,
        language=args.language,
        extraction_handler=extraction_handler,
        generation_handler=generation_handler,
        locomo=args.locomo,
    )

    print(f"记忆摘要完成，结果已写入: {out_file}")


if __name__ == "__main__":
    main()
