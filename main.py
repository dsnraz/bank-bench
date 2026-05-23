"""
MemoryBank-SiliconFriend 记忆摘要流水线（handler 模式）。

对标 hyperbolic_memory 的 session_run.py 架构:
  - Phase 1: 抽取模型 → 逐 session 摘要 + 人格分析
  - Phase 2: 生成模型 → 跨 session 整体汇总
  - Phase 3: locomo 模式下，建 FAISS 库 → 检索 → QA 预测

核心逻辑全部调用官方模块。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "memory_bank"))

from model_handler import create_model_handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MemoryBank-SiliconFriend 记忆摘要流水线"
    )

    # 数据路径
    p.add_argument("--memory-dir", type=str, required=True)
    p.add_argument("--out-file", type=str, default=None)
    p.add_argument("--user-name", type=str, default=None)
    p.add_argument("--language", type=str, default="en", choices=("cn", "en"))

    # 抽取模型（Phase 1: 逐日摘要 + 人格分析）
    p.add_argument("--extraction-handler-type", type=str, default="transformers",
                   choices=("transformers", "openai"))
    p.add_argument("--extraction-model-path", type=str, default=None)
    p.add_argument("--extraction-api-base", type=str, default="https://api.deepseek.com")
    p.add_argument("--extraction-api-key", type=str, default=None)

    # 生成模型（Phase 2/3: 整体汇总 + QA 预测）
    p.add_argument("--generation-handler-type", type=str, default="transformers",
                   choices=("transformers", "openai"))
    p.add_argument("--generation-model-path", type=str, default=None)
    p.add_argument("--generation-api-base", type=str, default="https://api.deepseek.com")
    p.add_argument("--generation-api-key", type=str, default=None)

    # 嵌入模型
    p.add_argument("--embedding-model", type=str,
                   default="sentence-transformers/all-mpnet-base-v2")

    # 模式
    p.add_argument("--locomo", action="store_true", default=False)

    # QA 预测 (locomo 模式)
    p.add_argument("--ann-file", type=str, default=None,
                   help="LoCoMo QA 标注文件（locomo10.json）")
    p.add_argument("--prediction-key", type=str, default="locomo_bank_prediction")
    p.add_argument("--vs-dir", type=str, default="/tmp/locomo_bank_vs",
                   help="FAISS 向量库持久化目录")
    p.add_argument("--qa-top-k", type=int, default=3)

    # 运行控制
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _load_handler(handler_type: str, model_path: str | None,
                  api_base: str, api_key: str | None,
                  device: str, label: str = ""):
    if handler_type == "transformers":
        if not model_path:
            print(f"[{label}] transformers 类型必须提供 model-path，跳过（回退到原 OpenAI 逻辑）")
            return None
        handler = create_model_handler("transformers")
        if not handler.load(model_path, device=device):
            raise RuntimeError(f"[{label}] 模型加载失败: {model_path}")
        return handler
    elif handler_type == "openai":
        model_name = model_path or "deepseek-chat"
        handler = create_model_handler("openai", api_base=api_base)
        if not handler.load(model_name, device=device, api_key=api_key):
            raise RuntimeError(f"[{label}] API 连接失败: {api_base}")
        return handler
    else:
        raise ValueError(f"未知 handler 类型: {handler_type}")


# ─── locomo 辅助 ────────────────────────────────────────────────────

def _flatten_locomo_sample(sample_id: str, sample_data: dict) -> dict:
    """将 locomo 嵌套结构展平为 bank 格式，供 JsonMemoryLoader 使用。"""
    sessions = sample_data.get("sessions", {})
    flat: Dict[str, Any] = {
        "history": {},
        "summary": {},
        "personality": {},
        "overall_history": sample_data.get("overall_history", ""),
        "overall_personality": sample_data.get("overall_personality", []),
    }
    for skey, entry in sessions.items():
        for date, turns in entry.get("history", {}).items():
            key = f"{skey}|{date}"
            normalized = []
            for pair in turns:
                values = list(pair.values())
                if len(values) >= 2 and values[0] and values[1]:
                    normalized.append({"query": values[0], "response": values[1]})
            if normalized:
                flat["history"][key] = normalized
        for date, s in entry.get("summary", {}).items():
            flat["summary"][f"{skey}|{date}"] = s
        for date, p in entry.get("personality", {}).items():
            flat["personality"][f"{skey}|{date}"] = p
    return flat


def _build_faiss(args) -> None:
    """Locomo 建库：每个 sample 展平后建一个 FAISS 索引，持久化到 --vs-dir。"""
    from memory_bank.memory_retrieval.local_doc_qa import LocalMemoryRetrieval

    with open(args.out_file, "r", encoding="utf-8") as f:
        bank = json.load(f)

    local_memory_qa = LocalMemoryRetrieval()
    local_memory_qa.init_cfg(
        embedding_model=args.embedding_model,
        embedding_device=args.device,
        top_k=args.qa_top_k,
        language=args.language,
    )

    vs_root = Path(args.vs_dir)
    vs_root.mkdir(parents=True, exist_ok=True)

    for sample_id, sample_data in bank.items():
        if args.user_name and sample_id != args.user_name:
            continue
        vs_path = str(vs_root / sample_id)
        if Path(vs_path, "index.faiss").exists():
            print(f"  FAISS 已存在，跳过: {vs_path}")
            continue

        flat = _flatten_locomo_sample(sample_id, sample_data)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump({sample_id: flat}, tmp, ensure_ascii=False)
            tmp_path = tmp.name
        vs_path, _ = local_memory_qa.init_memory_vector_store(
            filepath=tmp_path, vs_path=vs_path, user_name=sample_id
        )
        os.unlink(tmp_path)
        print(f"  FAISS 已建立: {vs_path}")


def _phase3_qa(args, generation_handler) -> None:
    """Phase 3: 加载 FAISS → 检索 → QA 预测。调用官方 LocalMemoryRetrieval。"""
    from memory_bank.memory_retrieval.local_doc_qa import LocalMemoryRetrieval

    with open(args.out_file, "r", encoding="utf-8") as f:
        bank = json.load(f)
    with open(args.ann_file, "r", encoding="utf-8") as f:
        ann_samples = json.load(f)

    ann_by_id = {s["sample_id"]: s for s in ann_samples}

    local_memory_qa = LocalMemoryRetrieval()
    local_memory_qa.init_cfg(
        embedding_model=args.embedding_model,
        embedding_device=args.device,
        top_k=args.qa_top_k,
        language=args.language,
    )

    vs_root = Path(args.vs_dir)

    from utils.prompt_utils import (
        generate_meta_prompt_dict_locomo_qa,
        build_prompt_locomo_qa,
    )
    meta_prompt = generate_meta_prompt_dict_locomo_qa()[args.language]

    predictions = OrderedDict()

    for sample_id, sample_data in bank.items():
        if args.user_name and sample_id != args.user_name:
            continue
        ann = ann_by_id.get(sample_id)
        if not ann or not ann.get("qa"):
            print(f"  {sample_id}: 无 QA 数据，跳过")
            continue

        print(f"\nPhase 3 — {sample_id}")

        flat = _flatten_locomo_sample(sample_id, sample_data)
        vs_path = str(vs_root / sample_id)
        if not Path(vs_path, "index.faiss").exists():
            print(f"  向量库不存在，跳过: {vs_path}")
            continue
        vector_store = local_memory_qa.load_memory_index(vs_path)

        sessions = sample_data.get("sessions", {})
        speaker_a = speaker_b = ""
        for entry in sessions.values():
            names = entry.get("name", [])
            speaker_a = names[0] if len(names) > 0 else ""
            speaker_b = names[1] if len(names) > 1 else ""
            break

        sample_pred = {"sample_id": sample_id, "qa": [dict(q) for q in ann["qa"]]}
        for qi, item in enumerate(sample_pred["qa"]):
            question = str(item.get("question", "")).strip()
            if not question:
                continue

            prompt, _ = build_prompt_locomo_qa(
                question=question,
                user_memory=flat,
                user_name=sample_id,
                user_memory_index=vector_store,
                local_memory_qa=local_memory_qa,
                meta_prompt=meta_prompt,
                user_keyword="[|User|]",
                ai_keyword="[|AI|]",
                boot_actual_name="",
                language=args.language,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
            )

            if generation_handler:
                answer = generation_handler.generate(prompt, max_new_tokens=32)
                item[args.prediction_key] = answer.strip() if answer else ""
            else:
                item[args.prediction_key] = ""

            print(f"  Q{qi+1}: {question[:80]}... → {item[args.prediction_key][:80]}")

        predictions[sample_id] = sample_pred

    pred_path = Path(args.out_file).parent / (
        Path(args.out_file).stem + "_qa_pred.json"
    )
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(list(predictions.values()), f, ensure_ascii=False, indent=2)
    print(f"\nQA 预测结果已写入: {pred_path}")


# ─── 主流程 ──────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1. 加载抽取 handler
    extraction_handler = _load_handler(
        args.extraction_handler_type, args.extraction_model_path,
        args.extraction_api_base, args.extraction_api_key,
        args.device, label="抽取模型"
    )

    # 2. 加载生成 handler（和抽取不同时）
    generation_handler = None
    if args.generation_handler_type != args.extraction_handler_type or \
       args.generation_model_path != args.extraction_model_path:
        generation_handler = _load_handler(
            args.generation_handler_type, args.generation_model_path,
            args.generation_api_base, args.generation_api_key,
            args.device, label="生成模型"
        )

    # 3. 设置嵌入模型
    from memory_bank.memory_retrieval.configs import model_config
    model_config.EMBEDDING_MODEL_EN = args.embedding_model
    print(f"[嵌入模型] {args.embedding_model}")

    # 4. Phase 1+2: 调用官方模块 summarize_memory
    from memory_bank.summarize_memory import summarize_memory

    memory_dir = args.memory_dir
    out_file = args.out_file or memory_dir
    print(f"输入记忆库: {memory_dir}")
    print(f"输出文件:   {out_file}")

    if out_file != memory_dir:
        import shutil
        shutil.copy(memory_dir, out_file)

    summarize_memory(
        out_file,
        name=args.user_name,
        language=args.language,
        extraction_handler=extraction_handler,
        generation_handler=generation_handler,
        locomo=args.locomo,
    )
    print("Phase 1+2 完成.")

    # 5. 建库: locomo 模式下每个 sample 建 FAISS（与 Phase 2 同级，仅一次）
    if args.locomo:
        _build_faiss(args)

    # 6. Phase 3: locomo 模式下 QA 预测
    if args.locomo and args.ann_file:
        if generation_handler is None:
            generation_handler = _load_handler(
                args.generation_handler_type, args.generation_model_path,
                args.generation_api_base, args.generation_api_key,
                args.device, label="生成模型(Phase 3)"
            )
        _phase3_qa(args, generation_handler)
        print("Phase 3 完成.")

    print(f"全部完成，结果已写入: {out_file}")


if __name__ == "__main__":
    main()
