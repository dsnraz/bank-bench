"""
将 LoCoMo locomo10.json 转换为 MemoryBank 格式。

层次关系:
  sample_1 (大样本)
    ├── session_1 → {name, history: {date_time: [...]}, summary, personality, overall_*}
    ├── session_2 → {name, history: {date_time: [...]}, summary, personality, overall_*}
    └── ...

每个 session 是一个完整的 bank 条目，外层按 10 个大样本分组。
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path


def build_dialogue_text(turn: dict) -> str:
    parts = [turn.get("text", "").strip()]
    caption = turn.get("blip_caption", "").strip()
    if caption:
        parts.append(f" [Shared image: {caption}]")
    query = turn.get("query", "").strip()
    if query:
        parts.append(f" [Image query: {query}]")
    img_desc = turn.get("image_description", "").strip()
    if img_desc:
        parts.append(f" [Image description: {img_desc}]")
    return " ".join(p for p in parts if p)


def make_session_entry(turns: list, date_time: str, speaker_a: str, speaker_b: str) -> dict:
    """将一个 session 的对话轮次转为 bank 条目。"""
    pairs = []
    i = 0
    while i < len(turns):
        a_turn = turns[i]
        a_text = build_dialogue_text(a_turn)
        pair = {a_turn.get("speaker", ""): a_text}

        if i + 1 < len(turns):
            b_turn = turns[i + 1]
            b_text = build_dialogue_text(b_turn)
            pair[b_turn.get("speaker", "")] = b_text
        pairs.append(pair)
        i += 2

    return {
        "name": [speaker_a, speaker_b],
        "history": {date_time: pairs},
        "summary": {},
        "personality": {},
        "overall_history": "",
        "overall_personality": "",
    }


def main():
    locomo_path = Path("/share/home/leiyh5/locomo/data/locomo10.json")
    if not locomo_path.is_file():
        if len(sys.argv) > 1:
            locomo_path = Path(sys.argv[1])
        else:
            print(f"未找到 locomo10.json: python {__file__} /path/to/locomo10.json")
            sys.exit(1)

    print(f"读取: {locomo_path}")
    with open(locomo_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"共 {len(samples)} 个样本")

    bank = OrderedDict()
    total_sessions = 0

    for sample in samples:
        sid = sample.get("sample_id", f"sample_{len(bank)}")
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "SpeakerA")
        speaker_b = conv.get("speaker_b", "SpeakerB")

        session_nums = sorted(
            int(k.split("_")[-1])
            for k in conv
            if k.startswith("session_") and "date_time" not in k
        )

        sample_entries = OrderedDict()
        for n in session_nums:
            turns = conv.get(f"session_{n}", [])
            date_time = conv.get(f"session_{n}_date_time", f"Session {n}")
            sample_entries[f"session_{n}"] = make_session_entry(
                turns, date_time, speaker_a, speaker_b
            )
            total_sessions += 1

        bank[sid] = sample_entries
        print(f"  {sid}: {len(sample_entries)} sessions")

    print(f"共 {total_sessions} 个 session")

    out_dir = Path(__file__).parent.parent / "resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "locomo10_bank.json"

    print(f"写入: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print("完成.")


if __name__ == "__main__":
    main()
