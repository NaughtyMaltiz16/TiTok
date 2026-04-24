# =============================================================================
# selective_train_diff_tokenizer_ver.py
#
# Trains a Llama-3.1-8B-Instruct student model using output-aligned excess loss
# masks produced by tokenizer_align.py (Mistral -> Llama3 alignment).
#
# Input: tokenizer_align.py outputs at
#   ./results/tokenizer_aligned/query_{queries_from}_EXPERT_{teacher_short}_on_target_{student_short}/
#     {task_name}/k{k_percent}_aligned_{student_short}/
#       {task_name}_output_aligned_{student_short}_k{k_percent}.pt
#
# Output: LoRA adapters saved to
#   ./checkpoints/aligned/{teacher_short}_{student_short}/{task_name}/k{k_percent}/
#
# Only supports Mistral (teacher) -> Llama-3.1-8B-Instruct (student).
#
# Sample command:
#   python selective_train_diff_tokenizer_ver.py \
#       --teacher_model mistralai/Mistral-7B-Instruct-v0.3 \
#       --student_model meta-llama/Llama-3.1-8B-Instruct \
#       --queries_from EXPERT_Mistral-7B-Instruct-v0.3 \
#       --task_name word_sorting \
#       --is_train \
#       --single_k 0.7 \
#       --seed 42
# =============================================================================

import math
import os
import json
import argparse
import shutil
import logging
import random
from tqdm import tqdm
import wandb
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict

from datasets import Dataset, disable_progress_bar
import transformers
from transformers import (
    set_seed, AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

disable_progress_bar()
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Hardcoded chat templates — student is always Llama3
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

LLAMA_CHAT_TEMPLATE = """
{% set bos_token = '<|begin_of_text|>' %}
{% set start_header = '<|start_header_id|>' %}
{% set end_header = '<|end_header_id|>' %}
{% set eot = '<|eot_id|>' %}

{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
    {{ start_header }}system{{ end_header }}
    {{ messages[0]['content'].strip() }}{{ eot }}
    {% set loop_messages = messages[1:] %}
{% else %}
    {% set loop_messages = messages %}
{% endif %}

{% for message in loop_messages %}
    {{ start_header }}{{ message['role'] }}{{ end_header }}
    {{ message['content'].strip() }}{{ eot }}
{% endfor %}
{{ start_header }}assistant{{ end_header }}
"""

# ---------------------------------------------------------------------------
# K values to process
# ---------------------------------------------------------------------------
K_VALUES = [0.7]

# ---------------------------------------------------------------------------
# BBH tasks
# ---------------------------------------------------------------------------
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Selective Training with Output-Aligned Excess Losses")
parser.add_argument('--student_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
parser.add_argument('--teacher_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
parser.add_argument('--queries_from', type=str, default='EXPERT_Mistral-7B-Instruct-v0.3')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--cut_off', type=int, default=768)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--task_name', type=str, default='word_sorting')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--r', type=int, default=8)
parser.add_argument('--alpha', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--single_task', type=str, default=None)
parser.add_argument('--single_k', type=float, default=None)
parser.add_argument('--continue_on_error', action='store_true')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

# ---------------------------------------------------------------------------
# System prompt — loaded from prompts.json
# ---------------------------------------------------------------------------
def load_prompts(path="./prompts.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_system_prompt(task_name, prompts):
    return prompts.get(task_name, prompts["default"])

PROMPTS = load_prompts("./prompts.json")

# ---------------------------------------------------------------------------
# AlignGroupBatchSampler — keeps training batches == aligner batches
# ---------------------------------------------------------------------------
class AlignGroupBatchSampler(Sampler):
    """Yield indices grouped by align_group_id so training batches == aligner batches."""
    def __init__(self, dataset, drop_last=False, shuffle=False, generator=None):
        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            gid = int(dataset[idx].get("align_group_id", -1))
            self.groups[gid].append(idx)
        self.gids = sorted(self.groups.keys())
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        gids = list(self.gids)
        if self.shuffle:
            g = self.generator if self.generator is not None else torch.Generator()
            order = torch.randperm(len(gids), generator=g).tolist()
            gids = [gids[i] for i in order]
        for gid in gids:
            batch = self.groups[gid]
            if not batch and self.drop_last:
                continue
            yield batch

    def __len__(self):
        return sum(1 for gid in self.gids if self.groups[gid] or not self.drop_last)

# ---------------------------------------------------------------------------
# Selective Trainer
# ---------------------------------------------------------------------------
class SelectiveTrainer(Trainer):
    def __init__(self, k_percent=0.7, wandb_config=None, **kwargs):
        self.k_percent = k_percent
        self.wandb_config = wandb_config or {}
        super().__init__(**kwargs)
        self.selected_ratios = []
        self.training_losses = []
        self.excess_losses = []
        self.selected_token_ids = []
        self.selected_token_positions = []
        self.positive_excess_count = 0
        self.negative_excess_count = 0
        self.total_tokens_processed = 0
        self.input_tokens_excluded = 0
        self.output_tokens_processed = 0

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        batch_sampler = AlignGroupBatchSampler(
            self.train_dataset,
            drop_last=self.args.dataloader_drop_last,
            shuffle=False,
            generator=None,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        if 'excess_losses' in inputs:
            return self._compute_selective_loss(model, inputs, return_outputs)
        else:
            return self._compute_standard_loss(model, inputs, return_outputs)

    def _compute_standard_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        labels = inputs.get('labels', input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _compute_selective_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs.get('labels', input_ids)
        excess_losses = inputs['excess_losses']  # [B, L-1] 0/1 mask

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1)).view(targets.shape)

        output_mask = (targets != -100)
        attend = attention_mask[:, 1:].contiguous().bool()
        pre_mask = (excess_losses > 0.5).bool()

        with torch.no_grad():
            total_attended = attend.sum().item()
            output_attended = (output_mask & attend).sum().item()
            self.output_tokens_processed += output_attended
            self.input_tokens_excluded += max(0, total_attended - output_attended)

        valid_loss_mask = output_mask & attend & pre_mask
        if valid_loss_mask.any():
            selected_losses = token_losses[valid_loss_mask]
            denom = max(1, valid_loss_mask.sum().item())
            selective_loss = selected_losses.sum() / denom
            actual_ratio = valid_loss_mask.sum().item() / max(1, (output_mask & attend).sum().item())
            self.selected_ratios.append(actual_ratio)
            self.training_losses.append(selective_loss.item())

            class SelectiveOutputs:
                def __init__(self, loss, logits): self.loss, self.logits = loss, outputs.logits
            return (selective_loss, SelectiveOutputs(selective_loss, outputs.logits)) if return_outputs else selective_loss
        else:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

# ---------------------------------------------------------------------------
# Dataset — fresh conversations with aligned losses
# ---------------------------------------------------------------------------
class FreshConversationDataset(torch.utils.data.Dataset):
    def __init__(self, aligned_output_data, student_tokenizer, task_name, student_model):
        self.student_tokenizer = student_tokenizer
        self.data = []

        # Student is always Llama3
        self.student_tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

        system_prompt = get_system_prompt(task_name, PROMPTS)
        aligned_samples = aligned_output_data['aligned_output_samples']

        for sample in aligned_samples:
            try:
                conversation_item = self._create_fresh_conversation_with_aligned_losses(
                    sample, system_prompt
                )
                if conversation_item:
                    self.data.append(conversation_item)
            except Exception as e:
                continue

        print(f"[FreshDataset] Created {len(self.data)} fresh conversation samples")

    def _create_fresh_conversation_with_aligned_losses(self, aligned_sample, system_prompt):
        # 1) Build full conversation with student chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": aligned_sample['input_text']},
            {"role": "assistant", "content": aligned_sample['output_text']},
        ]
        convo_text = self.student_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.student_tokenizer(convo_text, add_special_tokens=False)
        input_ids = torch.tensor(enc["input_ids"])
        attention_mask = torch.tensor(enc["attention_mask"])

        # 2) Build labels: mask everything up to assistant content
        input_only_text = self.student_tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        input_only_ids = self.student_tokenizer.encode(input_only_text, add_special_tokens=False)
        boundary = len(input_only_ids)

        labels = input_ids.clone()
        labels[:boundary] = -100

        # 3) Place the precomputed selection over assistant tokens
        mask_from_alignment = np.asarray(aligned_sample['target_binary_mask'], dtype=np.int32)

        ass_len = len(input_ids) - boundary
        assert ass_len >= 0, "Negative assistant length — boundary miscomputed"

        if len(mask_from_alignment) != ass_len:
            if 'warned_len' not in getattr(self, '__dict__', {}):
                self.warned_len = True
            m = min(len(mask_from_alignment), ass_len)
            tmp = np.zeros(ass_len, dtype=np.int32)
            if m > 0:
                tmp[:m] = mask_from_alignment[:m]
            mask_from_alignment = tmp

        # 4) Build excess_losses (length = seq_len - 1)
        excess_losses = np.zeros(len(input_ids) - 1, dtype=np.float32)
        out_mask = (labels[1:] != -100).numpy()
        start = max(0, boundary - 1)
        end = min(start + len(mask_from_alignment), len(excess_losses))
        put_len = max(0, end - start)
        if put_len > 0:
            excess_losses[start:end] = mask_from_alignment[:put_len]
        excess_losses = excess_losses * out_mask.astype(np.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "excess_losses": torch.tensor(excess_losses, dtype=torch.float32),
            "align_group_id": int(aligned_sample.get("align_group_id", -1)),
            "debug_info": {
                "sample_idx": aligned_sample.get("sample_idx"),
                "boundary": boundary,
                "assistant_len": ass_len,
                "excess_len": len(excess_losses),
                "selected_sum": float(excess_losses.sum()),
                "conversation_text": convo_text[:2000],
            },
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
def simple_data_collator(tokenizer, max_length=2048):
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        excess_losses = [item['excess_losses'] for item in batch]
        labels = [item['labels'] for item in batch]
        max_len = min(max(len(ids) for ids in input_ids), max_length)
        padded_input_ids = []
        padded_attention_mask = []
        padded_excess_losses = []
        padded_labels = []
        for i in range(len(batch)):
            ids = input_ids[i]
            mask = attention_mask[i]
            excess = excess_losses[i]
            lbls = labels[i]
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
                lbls = lbls[:max_len]
                if len(excess) >= max_len:
                    excess = excess[:max_len - 1]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=lbls.dtype)])
            target_excess_len = max_len - 1
            if len(excess) > target_excess_len:
                excess = excess[:target_excess_len]
            elif len(excess) < target_excess_len:
                excess = torch.cat([excess, torch.zeros(target_excess_len - len(excess), dtype=excess.dtype)])
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_excess_losses.append(excess)
            padded_labels.append(lbls)
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'excess_losses': torch.stack(padded_excess_losses),
            'labels': torch.stack(padded_labels),
        }
    return collate_fn

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name, access_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    # Student is always Llama3
    tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, token=access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

# ---------------------------------------------------------------------------
# Path resolution — compatible with tokenizer_align.py output
# ---------------------------------------------------------------------------
def find_aligned_output_data_path(task_name, k_percent, teacher_model, student_model):
    teacher_short = extract_model_name(teacher_model)
    student_short = extract_model_name(student_model)
    path = (
        f"./results/tokenizer_aligned"
        f"/query_{args.queries_from}_EXPERT_{teacher_short}_on_target_{student_short}"
        f"/{task_name}"
        f"/k{k_percent}_aligned_{student_short}"
        f"/{task_name}_output_aligned_{student_short}_k{k_percent}.pt"
    )
    if os.path.exists(path):
        return path
    print(f"Aligned data not found: {path}")
    return None

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_single_k(task_name, k_percent, teacher_model, student_model, training_args):
    print(f"\n[Training] Task: {task_name}, K: {k_percent}")

    aligned_path = find_aligned_output_data_path(task_name, k_percent, teacher_model, student_model)
    if not aligned_path:
        raise FileNotFoundError(f"Aligned data not found for {task_name} k={k_percent}")

    base_model, tokenizer = load_model_and_tokenizer(
        model_name=student_model, access_token=args.access_token
    )
    print_trainable_parameters(base_model)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    user_model = get_peft_model(base_model, lora_config)

    print(f"Loading aligned data from: {aligned_path}")
    aligned_data = torch.load(aligned_path, map_location='cpu', weights_only=False)

    # Verify model match
    actual_target = aligned_data.get('target_model', '')
    if student_model != actual_target:
        print(f"WARNING: Model mismatch! Expected {student_model}, got {actual_target}. Proceeding anyway.")

    dataset = FreshConversationDataset(aligned_data, tokenizer, task_name, student_model)

    if len(dataset) == 0:
        raise ValueError("Empty dataset")

    data_collator = simple_data_collator(tokenizer, max_length=args.cut_off)

    output_dir = f'./temp_outputs/{task_name}_k{k_percent}'
    training_args_obj = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim='adamw_torch',
        num_train_epochs=args.max_epoch,
        save_strategy="no",
        save_steps=1e10,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        group_by_length=False,
        neftune_noise_alpha=5,
        lr_scheduler_type='linear',
        report_to=['wandb'] if wandb.run is not None else [],
        remove_unused_columns=False,
    )

    for name, module in user_model.named_modules():
        if "norm" in name:
            module.to(user_model.dtype)

    trainer = SelectiveTrainer(
        model=user_model,
        args=training_args_obj,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        k_percent=k_percent,
    )

    trainer.train()

    teacher_short = extract_model_name(teacher_model)
    student_short = extract_model_name(student_model)
    save_path = f"./checkpoints/aligned/{teacher_short}_{student_short}/{task_name}/k{k_percent}"
    os.makedirs(save_path, exist_ok=True)
    user_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print(f"[Training Complete] Model saved to: {save_path}")
    print(f"  Input tokens excluded:  {trainer.input_tokens_excluded}")
    print(f"  Output tokens processed: {trainer.output_tokens_processed}")

    del user_model, base_model, trainer, dataset
    torch.cuda.empty_cache()

    return save_path

def train_task_all_k(task_name, teacher_model, student_model, training_args):
    results = {}
    successful = 0
    failed = 0
    k_values = K_VALUES if args.single_k is None else [args.single_k]

    for k_percent in k_values:
        try:
            print(f"\n{'='*60}")
            print(f"[{task_name}] Training k={k_percent}...")
            save_path = train_single_k(task_name, k_percent, teacher_model, student_model, training_args)
            results[k_percent] = {'status': 'success', 'save_path': save_path}
            successful += 1
            print(f"[{task_name}] k={k_percent} completed.")
        except Exception as e:
            print(f"[{task_name}] k={k_percent} failed: {e}")
            results[k_percent] = {'status': 'failed', 'error': str(e)}
            failed += 1
            if not args.continue_on_error:
                break

    print(f"\n[{task_name}] Summary: {successful}/{len(k_values)} k-values successful")
    return results

def train_all_tasks():
    teacher_model = args.teacher_model
    student_model = args.student_model
    tasks = BBH_TASKS if args.single_task is None else [args.single_task]

    print(f"\n{'='*80}")
    print(f"BBH SELECTIVE TRAINING — Mistral -> Llama-3.1-8B")
    print(f"{'='*80}")
    print(f"Teacher: {teacher_model}")
    print(f"Student: {student_model}")
    print(f"Tasks:   {len(tasks)}")
    print(f"K values: {K_VALUES if args.single_k is None else [args.single_k]}")
    print(f"{'='*80}")

    overall_results = {}
    task_success = 0
    task_fail = 0

    for idx, task in enumerate(tasks, 1):
        try:
            print(f"\n[{idx}/{len(tasks)}] Task: {task}")
            results = train_task_all_k(task, teacher_model, student_model, args)
            overall_results[task] = results
            successful_k = sum(1 for r in results.values() if r['status'] == 'success')
            if successful_k > 0:
                task_success += 1
            else:
                task_fail += 1
        except Exception as e:
            print(f"[{idx}/{len(tasks)}] {task} failed: {e}")
            overall_results[task] = {'error': str(e)}
            task_fail += 1
            if not args.continue_on_error:
                break

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"Tasks successful: {task_success}/{len(tasks)}  |  Failed: {task_fail}")
    for task, results in overall_results.items():
        if isinstance(results, dict) and 'error' not in results:
            successful_k = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'success')
            print(f"  {task}: {successful_k}/{len(results)} k-values")
        else:
            print(f"  {task}: Failed")
    print(f"{'='*80}")
    return overall_results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"BBH Selective Training — Mistral -> Llama-3.1-8B")
    print(f"Student: {args.student_model}")
    print(f"Teacher: {args.teacher_model}")
    set_all_seeds(args.seed)

    if args.is_train:
        train_all_tasks()
    else:
        print("Add --is_train flag to start training.")

if __name__ == "__main__":
    main()