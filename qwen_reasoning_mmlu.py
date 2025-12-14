import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict

model_path = "/home/jake0360/BAGEL/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

print("Loading MMLU")
dataset = load_dataset("cais/mmlu", "all", split="test")

label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
subject_results = defaultdict(lambda: {"correct": 0, "total": 0})

def format_prompt(ex):
    choices_text = f"A. {ex['choices'][0]}\nB. {ex['choices'][1]}\nC. {ex['choices'][2]}\nD. {ex['choices'][3]}"
    prompt_content = (
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {ex['question']}\n"
        f"{choices_text}\n\n"
        f"Give only the letter of the correct answer (A, B, C, or D)."
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You answer multiple choice questions. You output ONLY the correct letter."},
        {"role": "user", "content": prompt_content}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Starting Evaluation...")

for ex in tqdm(dataset, desc="Processing"):
    subject = ex["subject"]
    prompt = format_prompt(ex)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # set max token to 1
        output = model.generate(
            **inputs,
            max_new_tokens=1,
        )
    
    answer = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    correct_label = label_map[ex["answer"]]
    
    if answer == correct_label:
        subject_results[subject]["correct"] += 1
    subject_results[subject]["total"] += 1

print("\n=== PER-SUBJECT ACCURACY ===")
total_acc_sum = 0
num_subjects = 0

for subject in sorted(subject_results.keys()):
    stats = subject_results[subject]
    acc = stats["correct"] / stats["total"]
    total_acc_sum += acc
    num_subjects += 1
    print(f"{subject}: {acc:.2%} ({stats['correct']}/{stats['total']})")

print("-" * 30)
if num_subjects > 0:
    avg = total_acc_sum / num_subjects
    print(f"Final MMLU Score: {avg:.2%}")