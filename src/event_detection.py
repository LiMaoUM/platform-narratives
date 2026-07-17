# classify_events_vllm.py
import json
import math
import os
from typing import List, Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from vllm import LLM, SamplingParams

# -------- Config --------
# Set CUDA_VISIBLE_DEVICES in your environment to pick a GPU.
CSV_PATH = os.environ.get(
    "EVENT_DETECTION_CSV",
    "data/data/narrative_analysis_id_root_post_annotated_clean.csv",
)
TEXT_COL = "post_full_content"
ID_COL_CANDIDATES = ["id", "post_id", "root_id"]
OUTPUT_PATH = os.path.splitext(CSV_PATH)[0] + "_classified_gemma3.csv"

MODEL_NAME = "google/gemma-3-27b-it"   # replace with full HF repo id if needed
BATCH_SIZE = 32
MAX_TOKENS = 256
TEMPERATURE = 0.1
TOP_P = 0.9
SEED = 42
# ------------------------

SYSTEM_PROMPT = """You are a careful political event tagger. Your job is to decide whether a given post is about any of the following events (multi-label allowed), with a strict temporal focus on **May–June 2024**:

EVENT SET (all anchored to May–June 2024):
1) Trump trial — criminal/civil trial or court proceedings involving Donald Trump (e.g., hush money case, classified documents case, election interference).
2) Hunter Biden trial — trials/court proceedings involving Hunter Biden (e.g., gun case, tax case).
3) Presidential election debate — U.S. general-election presidential or vice-presidential debates that occurred or were scheduled in May–June 2024 (NOT primary debates). If the post is generic political chatter (polls, rallies, opinions) rather than a debate event itself, do NOT select.

TEMPORAL FOCUS (critical):
- Classify as true only if the post is about these events as they occurred or were being discussed in the May–June 2024 timeframe.
- If the post clearly refers to another year (e.g., 2020, 2021, 2022, 2023, 2025) or a month outside May–June 2024, leave the labels false and set none_of_above=true.
- If no explicit dates are present, infer cautiously from context (e.g., “today’s hearing,” “this week’s debate”) only if it plausibly aligns with May–June 2024; otherwise, prefer none_of_above=true.
- Do not use outside knowledge; rely only on the text provided.

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "trump_trial": true/false,
  "hunter_biden_trial": true/false,
  "presidential_election_debate": true/false,
  "none_of_above": true/false,
  "rationale": "one sentence citing textual clues (include any temporal cue if present)",
  "confidence": 0.0-1.0
}

RULES:
- Multiple labels can be true.
- If any of the three labels is true, set none_of_above=false; if all three are false, set none_of_above=true.
- Output JSON only — no extra text."""


FEW_SHOT = [
    # Trump trial in May 2024
    {
        "role": "user",
        "content": "The hush money trial dominated headlines in May 2024 as witnesses testified about Trump’s involvement."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "trump_trial": True,
            "hunter_biden_trial": False,
            "presidential_election_debate": False,
            "none_of_above": False,
            "rationale": "Clearly references Trump’s hush money trial in May 2024.",
            "confidence": 0.94
        })
    },

    # Hunter Biden trial in June 2024
    {
        "role": "user",
        "content": "Jury deliberations in Hunter Biden’s gun trial stretched into mid-June 2024."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "trump_trial": False,
            "hunter_biden_trial": True,
            "presidential_election_debate": False,
            "none_of_above": False,
            "rationale": "Explicitly references Hunter Biden’s gun trial in June 2024.",
            "confidence": 0.95
        })
    },

    # Presidential debate in June 2024
    {
        "role": "user",
        "content": "Tonight’s presidential debate in late June 2024 will put both candidates on the same stage."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "trump_trial": False,
            "hunter_biden_trial": False,
            "presidential_election_debate": True,
            "none_of_above": False,
            "rationale": "Directly refers to a presidential debate in June 2024.",
            "confidence": 0.93
        })
    },

    # Generic politics (not an event)
    {
        "role": "user",
        "content": "Voters in May 2024 said the economy mattered more than anything else."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "trump_trial": False,
            "hunter_biden_trial": False,
            "presidential_election_debate": False,
            "none_of_above": True,
            "rationale": "General political opinion polling; not tied to a trial or debate.",
            "confidence": 0.88
        })
    },

    # Out-of-window debate (should be excluded)
    {
        "role": "user",
        "content": "The presidential debate in September 2024 shifted the polls dramatically."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "trump_trial": False,
            "hunter_biden_trial": False,
            "presidential_election_debate": False,
            "none_of_above": True,
            "rationale": "Debate is in September 2024, outside the May–June 2024 window.",
            "confidence": 0.92
        })
    }
]


class EventLabels(BaseModel):
    trump_trial: bool = Field(default=False)
    hunter_biden_trial: bool = Field(default=False)
    presidential_election_debate: bool = Field(default=False)
    none_of_above: bool
    rationale: str = Field(default="", min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("none_of_above")
    @classmethod
    def check_none_of_above(cls, v, info):
        # Enforce mutual exclusion rule at validation time
        data = info.data
        any_true = bool(data.get("trump_trial")) or bool(data.get("hunter_biden_trial")) or bool(data.get("presidential_election_debate"))
        if any_true and v:
            raise ValueError("none_of_above must be false when any label is true.")
        if not any_true and not v:
            raise ValueError("none_of_above must be true when all labels are false.")
        return v

def build_messages(post_text: str):
    post_text = (post_text or "").strip()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT)
    messages.append({
        "role": "user",
        "content": f"Post Text:\n{post_text}\n\nReturn STRICT JSON only."
    })
    return messages

def apply_chat_template(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def try_parse_json(s: str) -> Dict[str, Any]:
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        return json.loads(t)
    except Exception:
        try:
            start = t.find("{")
            end = t.rfind("}")
            return json.loads(t[start:end+1])
        except Exception:
            # Fallback default (will still be validated later)
            return {
                "trump_trial": False,
                "hunter_biden_trial": False,
                "presidential_election_debate": False,
                "none_of_above": True,
                "rationale": "Parse error; defaulted to none_of_above.",
                "confidence": 0.3
            }

def main():
    df = pd.read_csv(CSV_PATH)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column {TEXT_COL} not found. Available: {list(df.columns)}")
    id_col: Optional[str] = next((c for c in ID_COL_CANDIDATES if c in df.columns), None)

    llm = LLM(model=MODEL_NAME, seed=SEED,max_model_len=18192)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)

    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    total = len(texts)
    num_batches = math.ceil(total / BATCH_SIZE)

    rows = []
    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, total)
        batch = texts[start:end]

        prompts = [apply_chat_template(tokenizer, build_messages(t)) for t in batch]
        outputs = llm.generate(prompts, sampling_params=sampling)

        for i, out in enumerate(outputs):
            gen = out.outputs[0].text if out.outputs else ""
            raw = try_parse_json(gen)

            # Validate with Pydantic, fix simple rule violations if needed
            try:
                record = EventLabels.model_validate(raw)
            except ValidationError:
                # last-chance fix for the none_of_above rule
                any_true = bool(raw.get("trump_trial")) or bool(raw.get("hunter_biden_trial")) or bool(raw.get("presidential_election_debate"))
                raw["none_of_above"] = not any_true
                raw.setdefault("rationale", "Auto-corrected to satisfy schema rules.")
                raw.setdefault("confidence", 0.5)
                record = EventLabels.model_validate(raw)

            item = {
                "row_index": start + i,
                "trump_trial": record.trump_trial,
                "hunter_biden_trial": record.hunter_biden_trial,
                "presidential_election_debate": record.presidential_election_debate,
                "none_of_above": record.none_of_above,
                "rationale": record.rationale,
                "confidence": record.confidence,
                "raw_model_output": gen,
            }
            if id_col:
                item[id_col] = df.iloc[start + i][id_col]
            rows.append(item)

        print(f"Processed batch {b+1}/{num_batches}")

    out = pd.DataFrame(rows)
    if id_col:
        merged = df.merge(out.drop(columns=["row_index"]), on=id_col, how="left")
    else:
        merged = pd.concat([df, out.drop(columns=["row_index"])], axis=1)

    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
