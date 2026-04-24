# =============================================================================
#
# [SFT — Training]
# Fine-tunes a LLM on BBH tasks using LoRA (PEFT).
# Saves the LoRA adapter + tokenizer + training_config.json to ./checkpoints/.
#
# Sample command:
#   python SFT.py \
#       --is_train \
#       --task_name word_sorting \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --learning_rate 5e-5 \
#       --max_epoch 2 \
#       --seed 42
#
# [Inference]
# Loads the saved LoRA adapter (or the bare base model with --initial_model)
# and runs batched generation on the BBH test split.
# Results are saved to ./results/{task_name}/.
#
# Sample command:
#   python SFT.py \
#       --is_infer \
#       --task_name word_sorting \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --learning_rate 5e-5 \
#       --max_epoch 2 \
#       --seed 42
#
#   # Infer with base model only (no adapter):
#   python SFT.py \
#       --is_infer \
#       --task_name word_sorting \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --initial_model
# =============================================================================

import torch
import argparse
import json
import os
import logging
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from utils import split_batch, print_trainable_parameters, get_output

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="BBH SFT with LoRA")
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_step", type=int, default=5000)
parser.add_argument("--cut_off", type=int, default=768)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--max_epoch", type=int, default=2)
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--task_name", type=str, default="word_sorting")
parser.add_argument("--access_token", type=str, default=None)
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--alpha", type=int, default=8)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--initial_model", action="store_true",
                    help="Run inference with the base model (no adapter)")
parser.add_argument("--is_train", action="store_true")
parser.add_argument("--is_infer", action="store_true")
parser.add_argument("--seed", type=int, default=42)

"""
Example usage — run all 27 BBH tasks:

for task in boolean_expressions causal_judgement date_understanding \
            disambiguation_qa dyck_languages formal_fallacies geometric_shapes \
            hyperbaton logical_deduction_five_objects logical_deduction_seven_objects \
            logical_deduction_three_objects movie_recommendation multistep_arithmetic_two \
            navigate object_counting penguins_in_a_table reasoning_about_colored_objects \
            ruin_names salient_translation_error_detection snarks sports_understanding \
            temporal_sequences tracking_shuffled_objects_five_objects \
            tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects \
            web_of_lies word_sorting; do

    python SFT.py \
        --task_name ${task} \
        --is_train \
        --model_name mistralai/Mistral-7B-Instruct-v0.3 \
        --seed 0

done
"""

args = parser.parse_args()
task_name = args.task_name
batch_size = args.batch_size
cutoff_len = args.cut_off
max_epoch = args.max_epoch
model_name_short = args.model_name.split("/")[-1]
set_seed(args.seed)

# ---------------------------------------------------------------------------
# Chat templates (fallback if tokenizer has no built-in template)
# ---------------------------------------------------------------------------
MISTRAL_CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'].strip() + '\n' %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
"""

LLAMA_CHAT_TEMPLATE = (
    "{% set bos_token = '<|begin_of_text|>' %}\n"
    "{% set start_header = '<|start_header_id|>' %}\n"
    "{% set end_header = '<|end_header_id|>' %}\n"
    "{% set eot = '<|eot_id|>' %}\n\n"
    "{{ bos_token }}\n"
    "{% if messages[0]['role'] == 'system' %}\n"
    "    {{ start_header }}system{{ end_header }}\n"
    "    {{ messages[0]['content'].strip() }}{{ eot }}\n"
    "    {% set loop_messages = messages[1:] %}\n"
    "{% else %}\n"
    "    {% set loop_messages = messages %}\n"
    "{% endif %}\n\n"
    "{% for message in loop_messages %}\n"
    "    {{ start_header }}{{ message['role'] }}{{ end_header }}\n"
    "    {{ message['content'].strip() }}{{ eot }}\n"
    "{% endfor %}\n"
    "{{ start_header }}assistant{{ end_header }}"
)

# ---------------------------------------------------------------------------
# Prompt loading from prompts.json
# ---------------------------------------------------------------------------
def load_prompts(prompts_path: str = "./prompts.json") -> dict:
    with open(prompts_path, "r") as f:
        return json.load(f)

def get_system_prompt(task_name: str, prompts: dict) -> str:
    prompt = prompts.get(task_name, prompts["default"])
    print(f"System prompt for '{task_name}': '{prompt}'")
    return prompt

PROMPTS = load_prompts("./prompts.json")
SYSTEM_PROMPT = get_system_prompt(args.task_name, PROMPTS)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bbh_data(file_path: str) -> list:
    if not os.path.exists(file_path):
        print(f"Warning: Data file '{file_path}' not found. Returning empty list.")
        return []
    with open(file_path, "r") as f:
        data = json.load(f)
    examples = data.get("examples", [])
    print(f"Loaded {len(examples)} examples from {file_path}")
    return examples

train_data = load_bbh_data(f"./data/bbh_split/{args.task_name}_train.json")
infer_data = load_bbh_data(f"./data/bbh_split/{args.task_name}_test.json")

print("#" * 70)
logging.basicConfig(level=logging.INFO)
logging.info(f"Model: {args.model_name}")

# ---------------------------------------------------------------------------
# Model family detection + chat template application
# ---------------------------------------------------------------------------
def detect_model_family(model_name: str) -> str:
    lower = model_name.lower()
    if "mistral" in lower:
        return "mistral"
    elif "llama" in lower:
        return "llama3"
    else:
        raise ValueError(f"Unsupported model family for: {model_name}. Add a chat template for this model.")

CHAT_TEMPLATES = {
    "mistral": MISTRAL_CHAT_TEMPLATE,
    "llama3": LLAMA_CHAT_TEMPLATE,
}

def apply_chat_template(example, tokenizer):
    family = detect_model_family(args.model_name)
    tokenizer.chat_template = CHAT_TEMPLATES[family]
    print(f"Using {family} chat template")
    example["text_prompt"] = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    example["text_chosen"] = example["chosen"][0]["content"]
    return example

# ---------------------------------------------------------------------------
# Dataset construction

def create_bbh_dataset(train_data: list, tokenizer) -> list:
    sft_data = []
    print(f"Creating dataset with system prompt: '{SYSTEM_PROMPT}'")
    for item in train_data:
        example = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["input"]},
            ],
            "chosen": [{"role": "assistant", "content": item["target"]}],
        }
        fmt = apply_chat_template(example, tokenizer)
        sft_data.append({"prompt": fmt["text_prompt"], "chosen": fmt["text_chosen"]})
    if sft_data:
        print(f"Sample prompt:\n{sft_data[0]['prompt'][:200]}...")
        print(f"Sample chosen: {sft_data[0]['chosen']}")
    print(f"Created {len(sft_data)} training examples")
    return sft_data

def generate_and_tokenize_prompt(data_point):
    tokenized_prompt = tokenizer(
        data_point["prompt"],
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False,
    )
    tokenized_chosen = tokenizer(
        data_point["chosen"],
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False,
    )
    input_ids = tokenized_prompt["input_ids"] + tokenized_chosen["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"] + tokenized_chosen["attention_mask"]
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_chosen["input_ids"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------------------------------------------------------------------
# Training (SFT)
# ---------------------------------------------------------------------------
if args.is_train:
    if not train_data:
        print("Error: No training data found.")
        exit(1)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=args.access_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left", token=args.access_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    train_dataset_list = create_bbh_dataset(train_data, tokenizer)
    train_dataset = Dataset.from_list(train_dataset_list)
    print("Sample training example:")
    print(train_dataset[0])

    train_dataset = train_dataset.shuffle()
    train_dataset = train_dataset.map(
        generate_and_tokenize_prompt, remove_columns=["prompt", "chosen"]
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./outputs/",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        num_train_epochs=max_epoch,
        save_steps=1e9,
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        group_by_length=False,
        lr_scheduler_type="linear",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # Keep LayerNorm in model dtype
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(model.dtype)

    trainer.train()

    adapter_path = (
        f"./checkpoints/expert/{args.task_name}"
        f"/{args.learning_rate}-{args.max_epoch}-{model_name_short}"
    )
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Save training config for inference alignment
    training_config = {
        "model_name": args.model_name,
        "task_name": args.task_name,
        "system_prompt": SYSTEM_PROMPT,
        "training_params": {
            "learning_rate": args.learning_rate,
            "max_epoch": args.max_epoch,
            "batch_size": args.batch_size,
        },
    }
    with open(os.path.join(adapter_path, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"Adapter saved to: {adapter_path}")
    print("Training config saved for inference alignment.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
if args.is_infer:
    if not infer_data:
        print("Error: No inference data found.")
        exit(1)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=args.access_token,
    )
    base_model.config.use_cache = False

    if args.initial_model:
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, padding_side="left", token=args.access_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        infer_system_prompt = get_system_prompt(args.task_name, PROMPTS)
    else:
        adapter_path = (
            f"./checkpoints/expert/{args.task_name}"
            f"/{args.learning_rate}-{args.max_epoch}-{model_name_short}"
        )

        # Load training config to align system prompt with training
        config_path = os.path.join(adapter_path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                training_config = json.load(f)
            infer_system_prompt = training_config["system_prompt"]
            print(f"Loaded system prompt from training config: '{infer_system_prompt}'")
        else:
            print("Warning: training_config.json not found. Deriving system prompt from args.")
            infer_system_prompt = get_system_prompt(args.task_name, PROMPTS)

        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, padding_side="left", token=args.access_token
        )
        model = PeftModel.from_pretrained(
            model=base_model, model_id=adapter_path, is_trainable=False
        )

    model.eval()

    # Build prompt list
    infer_question_list = []
    question_list = []
    for item in tqdm(infer_data, desc="Preparing inference data"):
        chat = [
            {"role": "system", "content": infer_system_prompt},
            {"role": "user", "content": item["input"]},
        ]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        infer_question_list.append(formatted)
        question_list.append(item["input"])

    print(f"Sample inference prompt:\n{infer_question_list[0][:300]}...")

    # Batched generation
    out_list = []
    infer_batch_size = 16
    batch_list = split_batch(infer_question_list, infer_batch_size)

    with torch.no_grad():
        for batch in tqdm(batch_list, desc="Running inference"):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_token_type_ids=False,
            )
            inputs = inputs.to(model.device)
            with torch.autocast(device_type="cuda"):
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    top_k=50,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=200,
                    repetition_penalty=args.repetition_penalty,
                )
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            out_sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            out_list.extend([get_output(s, args) for s in out_sentences])

    # Collect and display results
    pred_all = []
    for i, output in enumerate(out_list):
        pred_all.append({
            "input": question_list[i],
            "output": output,
            "expected": infer_data[i]["target"],
        })
        print(f"Input:    {question_list[i]}")
        print(f"Output:   {output}")
        print(f"Expected: {infer_data[i]['target']}")
        print("-" * 50)

    # Build file suffix
    file_suffix = "-initial" if args.initial_model else ""

    # Save results
    output_file = {
        "examples": [
            {"input": item["input"], "target": item["output"]} for item in pred_all
        ],
        "model": args.model_name,
        "task": args.task_name,
        "system_prompt_used": infer_system_prompt,
    }

    result_path = (
        f"./results/{args.task_name}"
        f"/{args.learning_rate}-{args.max_epoch}-{model_name_short}{file_suffix}.json"
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(output_file, f, indent=2)

    print(f"Results saved to: {result_path}")
    print(f"System prompt used: '{infer_system_prompt}'")
    print("Inference complete.")