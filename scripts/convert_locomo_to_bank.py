"""
将 LoCoMo locomo10.json 转换为 MemoryBank 格式。

LoCoMo 结构:
  [{sample_id, conversation: {speaker_a, speaker_b, session_N: [{speaker, dia_id, text, ...}], session_N_date_time: ...}, qa: [...]}]

MemoryBank 结构:
  {sample_id: {name: [a, b], history: {date_time: [{a_name: "...", b_name: "..."}]}, summary: {}, personality: {}, overall_history: "", overall_personality: ""}}

特殊处理: 若对话轮次含有 blip_caption / query / image_description / image_query，拼入该角色的对话文本中。
"""

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


def build_dialogue_text(turn: dict) -> str:
    """将一轮对话的 text + 可选图片描述/查询拼成完整文本。"""
    parts = [turn.get("text", "").strip()]

    # blip_caption: 图片的自动描述
    caption = turn.get("blip_caption", "").strip()
    if caption:
        parts.append(f" [Shared image: {caption}]")

    # query: 图片相关的查询/描述
    query = turn.get("query", "").strip()
    if query:
        parts.append(f" [Image query: {query}]")

    # image_description: 直接图片描述字段 (部分数据可能有)
    img_desc = turn.get("image_description", "").strip()
    if img_desc:
        parts.append(f" [Image description: {img_desc}]")

    return " ".join(p for p in parts if p)


def convert_sample(sample: dict) -> dict:
    """将单个 LoCoMo sample 转为 MemoryBank 用户条目。"""
    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "SpeakerA")
    speaker_b = conv.get("speaker_b", "SpeakerB")

    # 收集所有 session 编号
    session_nums = sorted(
        int(k.split("_")[-1])
        for k in conv
        if k.startswith("session_") and "date_time" not in k
    )

    history = OrderedDict()
    for n in session_nums:
        session_key = f"session_{n}"
        date_key = f"session_{n}_date_time"
        date_time = conv.get(date_key, f"Session {n}")
        turns = conv.get(session_key, [])

        pairs = []
        i = 0
        while i < len(turns):
            a_turn = turns[i]
            a_speaker = a_turn.get("speaker", "")
            a_text = build_dialogue_text(a_turn)

            b_text = ""
            if i + 1 < len(turns):
                b_turn = turns[i + 1]
                b_speaker = b_turn.get("speaker", "")
                b_text = build_dialogue_text(b_turn)
            else:
                b_speaker = ""

            # 构建 {speaker_name: text} 对
            pair = {}
            pair[a_speaker] = a_text
            if b_text:
                pair[b_speaker] = b_text
            pairs.append(pair)
            i += 2

        history[date_time] = pairs

    return {
        "name": [speaker_a, speaker_b],
        "history": history,
        "summary": {},
        "personality": {},
        "overall_history": "",
        "overall_personality": "",
    }


def main():
    locomo_path = Path(__file__).parent.parent.parent / "locomo" / "data" / "locomo10.json"
    if not locomo_path.is_file():
        # 尝试通过命令行参数指定路径
        if len(sys.argv) > 1:
            locomo_path = Path(sys.argv[1])
        else:
            print(f"未找到 locomo10.json，请指定路径: python {__file__} /path/to/locomo10.json")
            sys.exit(1)

    print(f"读取: {locomo_path}")
    with open(locomo_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"共 {len(samples)} 个样本")

    bank = OrderedDict()
    for sample in samples:
        sid = sample.get("sample_id", f"sample_{len(bank)}")
        bank[sid] = convert_sample(sample)
        session_count = len(bank[sid]["history"])
        total_turns = sum(len(pairs) for pairs in bank[sid]["history"].values())
        print(f"  {sid}: {session_count} sessions, {total_turns} turn-pairs")

    # 输出到 resources 目录
    out_dir = Path(__file__).parent.parent / "resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "locomo10_bank.json"

    print(f"\n写入: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print("完成.")


if __name__ == "__main__":
    main()
