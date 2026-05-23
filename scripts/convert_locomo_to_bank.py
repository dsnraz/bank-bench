"""
将 LoCoMo locomo10.json 转换为 MemoryBank 格式。

LoCoMo:
  [{sample_id, conversation: {speaker_a, speaker_b, session_N: [{speaker, dia_id, text, ...}], session_N_date_time: ...}}]

MemoryBank: 每个 session 独立成一个条目
  {"{sample_id}__session_{N}": {name: [a, b], history: {date_time: [{a: "...", b: "..."}]}, summary: {}, personality: {}, overall_history: "", overall_personality: ""}}

只处理对话，忽略 QA。
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
    total_entries = 0

    for sample in samples:
        sid = sample.get("sample_id", f"sample_{len(bank)}")
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "SpeakerA")
        speaker_b = conv.get("speaker_b", "SpeakerB")

        # 收集所有 session 编号
        session_nums = sorted(
            int(k.split("_")[-1])
            for k in conv
            if k.startswith("session_") and "date_time" not in k
        )

        for n in session_nums:
            entry_key = f"{sid}__session_{n}"

            turns = conv.get(f"session_{n}", [])
            date_time = conv.get(f"session_{n}_date_time", f"Session {n}")

            # 两两配对
            pairs = []
            i = 0
            while i < len(turns):
                a_turn = turns[i]
                a_speaker = a_turn.get("speaker", "")
                a_text = build_dialogue_text(a_turn)

                pair = {a_speaker: a_text}
                if i + 1 < len(turns):
                    b_turn = turns[i + 1]
                    b_speaker = b_turn.get("speaker", "")
                    b_text = build_dialogue_text(b_turn)
                    pair[b_speaker] = b_text
                pairs.append(pair)
                i += 2

            bank[entry_key] = {
                "name": [speaker_a, speaker_b],
                "history": {date_time: pairs},
                "summary": {},
                "personality": {},
                "overall_history": "",
                "overall_personality": "",
            }
            total_entries += 1

    print(f"共 {total_entries} 个 session → bank 条目")

    out_dir = Path(__file__).parent.parent / "resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "locomo10_bank.json"

    print(f"写入: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print("完成.")


if __name__ == "__main__":
    main()
