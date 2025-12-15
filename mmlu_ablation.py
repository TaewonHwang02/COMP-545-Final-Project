#!/usr/bin/env python3
import os
import json
import argparse

import torch
from datasets import load_dataset

# Bagel-specific loader
from eval.vlm.utils import load_model_and_tokenizer


DEFAULT_SUBJECTS = [
    "abstract_algebra",
    "high_school_mathematics",
    "high_school_physics",
    "college_mathematics",
    "computer_security",
    "formal_logic",
]

def build_prompt(dev_set, q):
    shots = ""
    for ex in dev_set:
        choices = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip("ABCD", ex["choices"])
        )
        answer_letter = "ABCD"[ex["answer"]]
        shots += f"{ex['question']}\n{choices}\nAnswer: {answer_letter}\n\n"

    q_choices = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip("ABCD", q["choices"])
    )
    prompt = shots + f"{q['question']}\n{q_choices}\nAnswer: "
    return prompt



def predict_letter(model, tokenizer, new_token_ids, prompt):
   
    text = model.chat(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        image_transform=None,
        images=[],
        prompt=prompt,
        max_length=16,         
        do_sample=False,
        temperature=0.0,
    )

    for ch in text:
        if ch in "ABCD":
            return ch
    return None

def evaluate_subject(model, tokenizer, new_token_ids, subject, max_questions=None):
    """
    Evaluate one subject of MMLU (cais/mmlu).
    Uses dev as few-shot context and test as eval split.
    Respects HF_CACHE_DIR for offline use.
    """
    cache_dir = os.environ.get("HF_CACHE_DIR")
    if cache_dir:
        ds = load_dataset("cais/mmlu", subject, cache_dir=cache_dir)
    else:
        ds = load_dataset("cais/mmlu", subject)

    dev = ds["dev"]
    test = ds["test"]

    correct = 0
    total = len(test) if max_questions is None else min(max_questions, len(test))

    for i, example in enumerate(test):
        if max_questions is not None and i >= max_questions:
            break

        prompt = build_prompt(dev, example)
        pred = predict_letter(model, tokenizer, new_token_ids, prompt)
        gold = "ABCD"[example["answer"]]

        if pred == gold:
            correct += 1

        if (i + 1) % 20 == 0:
            print(f"[{subject}] {i+1}/{total} — acc = {correct/(i+1):.3f}")

    acc = correct / total
    return acc, correct, total


# =================== MAIN ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to BAGEL model directory (with llm_config/vit_config/ema.safetensors)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=",".join(DEFAULT_SUBJECTS),
        help="Comma-separated list of MMLU subjects to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store JSON results",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on number of test questions per subject",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read ablated layer from env (set by SLURM array)
    ablate_layer_str = os.environ.get("ABLATE_LAYER", "-1")
    try:
        ablate_layer = int(ablate_layer_str)
    except ValueError:
        ablate_layer = -1

    print(f">>> ABLATE_LAYER = {ablate_layer}")
    print(f">>> MODEL_PATH   = {args.model_path}")

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    # -------- Load BAGEL model via its own utils --------
    print(">>> Loading BAGEL model with eval.vlm.utils.load_model_and_tokenizer...")
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    model.eval()

    # -------- Evaluate subjects --------
    results = {}
    total_correct = 0
    total_questions = 0

    for subj in subjects:
        print(f"\n=== Evaluating subject: {subj} ===")
        acc, correct, total = evaluate_subject(
            model,
            tokenizer,
            new_token_ids,
            subj,
            max_questions=args.max_questions,
        )
        results[subj] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }

        total_correct += correct
        total_questions += total

        print(f"Subject {subj}: {acc * 100:.2f}% ({correct}/{total})")

    overall_acc = total_correct / total_questions if total_questions > 0 else 0.0

    # -------- Save results --------
    results_summary = {
        "ablate_layer": ablate_layer,
        "subjects": subjects,
        "per_subject": results,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_questions": total_questions,
    }

    out_path = os.path.join(
        args.output_dir,
        f"mmlu_layer_{ablate_layer}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n=== DONE ===")
    print(f"Overall accuracy: {overall_acc * 100:.2f}% "
          f"({total_correct}/{total_questions})")
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
