"""
Utility functions for ChartQA fine-tuning with Unsloth.

Used by: notebook.ipynb

Based on the original torchtune tutorial:
https://github.com/seungrokj/AAI25_workshop/blob/main/workshop_102/utils.py
"""

import json
import os

import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# 1. Dataset creation
# ---------------------------------------------------------------------------

def create_chart_qa_with_reasoning_dataset(reasoning_file, output_dir, override=False):
    """
    Merge ChartQA with synthetic chain-of-thought reasoning.

    The *reasoning_file* (``reasoning.parquet``) contains a ``label`` column
    with CoT reasoning text.  This function appends each reasoning string to
    the original ChartQA answer so the final label becomes
    ``"{reasoning} {original_answer}"``.

    Args:
        reasoning_file: Path to the parquet file with reasoning data.
        output_dir:     Directory to save the merged HF dataset.
        override:       If *True*, recreate even when *output_dir* exists.
    """
    if os.path.exists(output_dir) and not override:
        print(f"Dataset already exists at {output_dir}. "
              "Set override=True to recreate.")
        return

    if os.path.exists(output_dir) and override:
        import shutil
        shutil.rmtree(output_dir)

    print("Loading reasoning data …")
    reasoning_df = pd.read_parquet(reasoning_file)
    print(f"   columns : {reasoning_df.columns.tolist()}")
    print(f"   samples : {len(reasoning_df)}")

    print("Loading ChartQA from HuggingFace …")
    ds = load_dataset("HuggingFaceM4/ChartQA", split="train")
    print(f"   ChartQA train samples: {len(ds)}")

    reasoning_records = (
        pd.DataFrame({"reasoning": reasoning_df["label"]})
        .to_dict(orient="records")
    )

    ds = ds.map(
        lambda _, idx: {"reasoning": reasoning_records[idx]["reasoning"]},
        with_indices=True,
    )

    def _merge(row):
        original = row["label"]
        if isinstance(original, list):
            original = original[0]
        row["label"] = f"{row['reasoning']} {original}"
        return row

    ds = ds.map(_merge)
    ds = ds.remove_columns(["reasoning"])

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    ds.to_parquet(os.path.join(train_dir, "data.parquet"))

    with open(os.path.join(output_dir, "dataset_dict.json"), "w") as f:
        json.dump({"splits": ["train"]}, f, indent=4)

    print(f"Saved {len(ds)} samples -> {output_dir}")

    sample_idx = min(27901, len(ds) - 1)
    print(f"\nSample #{sample_idx}:")
    print(f"   query : {ds[sample_idx]['query']}")
    print(f"   label : {ds[sample_idx]['label'][:200]}…")


# ---------------------------------------------------------------------------
# 2. Formatting for text-only (Unsloth) training
# ---------------------------------------------------------------------------

def format_chartqa_for_text_training(dataset, max_samples=None):
    """
    Format ChartQA as **text-only** conversations for Unsloth / SFTTrainer.

    Returns an HF ``Dataset`` with a single ``conversations`` column::

        [ {"role": "user",      "content": "…"},
          {"role": "assistant", "content": "…"} ]

    Args:
        dataset:     HF dataset with ``query`` and ``label`` columns.
        max_samples: Cap the number of samples (``None`` = use all).
    """
    n = min(max_samples, len(dataset)) if max_samples else len(dataset)
    rows = []
    for idx in range(n):
        sample = dataset[idx]
        answer = sample["label"]
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        rows.append({
            "conversations": [
                {
                    "role": "user",
                    "content": (
                        "Look at this chart and answer the following "
                        f"question.\n\nQuestion: {sample['query']}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]
        })
    ds = Dataset.from_list(rows)
    print(f"Formatted {len(ds)} text-only training samples")
    return ds


# ---------------------------------------------------------------------------
# 3. Formatting for vision (Unsloth FastVisionModel) training
# ---------------------------------------------------------------------------

def format_chartqa_for_vision_training(
    original_dataset,
    reasoning_dataset,
    max_samples=None,
):
    """
    Format ChartQA as **multimodal** conversations for Unsloth FastVisionModel.

    Each sample carries the actual PIL image from ChartQA alongside
    the text question and CoT answer.

    Returns a *list* of dicts with keys ``messages`` and ``images``
    (convert to ``Dataset.from_list(...)`` afterwards).

    Args:
        original_dataset:  HF ChartQA split (contains ``image`` column).
        reasoning_dataset: CoT dataset (contains ``label`` column).
        max_samples:       Cap the number of samples.
    """
    n = (min(max_samples, len(original_dataset))
         if max_samples else len(original_dataset))
    
    print(f"Formatting {n} samples for vision training...", flush=True)
    
    data, skipped = [], 0
    batch_size = 100  # Process in batches for better I/O efficiency
    
    # Use tqdm for progress bar
    pbar = tqdm(total=n, desc="Formatting", unit="samples")
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_indices = list(range(batch_start, batch_end))
        
        # Batch select from datasets (more efficient than single access)
        orig_batch = original_dataset.select(batch_indices)
        reason_batch = reasoning_dataset.select(batch_indices)
        
        for i in range(len(batch_indices)):
            image = orig_batch[i].get("image")
            query = orig_batch[i]["query"]
            answer = reason_batch[i]["label"]
            
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            if image is None:
                skipped += 1
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Format for Unsloth FastVisionModel
            data.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": (
                                    "Look at this chart and answer the "
                                    "following question.\n\n"
                                    f"Question: {query}"
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    },
                ],
            })
        
        pbar.update(batch_end - batch_start)
    
    pbar.close()
    print(f"✓ Formatted {len(data)} vision samples "
          f"({skipped} skipped — missing images)", flush=True)
    return data


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def compute_chartqa_metrics(predictions, references):
    """
    Compute ChartQA accuracy metrics.

    Returns a dict with ``exact_match`` and ``relaxed_accuracy``
    (answer appears anywhere in the prediction).
    """
    exact, relaxed = 0, 0
    for pred, ref in zip(predictions, references):
        p, r = str(pred).lower().strip(), str(ref).lower().strip()
        if p == r:
            exact += 1
            relaxed += 1
        elif r in p:
            relaxed += 1
    n = len(predictions) or 1
    return {"exact_match": exact / n, "relaxed_accuracy": relaxed / n}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("utils_unsloth.py loaded — functions available:")
    print("  create_chart_qa_with_reasoning_dataset")
    print("  format_chartqa_for_text_training")
    print("  format_chartqa_for_vision_training")
    print("  compute_chartqa_metrics")
