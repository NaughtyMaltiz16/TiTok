# =============================================================================
# selective_train_same_tokenizer_ver.py
#
# Selective training using precomputed excess losses.
# Reads top_{k}_tensors.pt and all_excess_losses.pt produced by
# compute_contrastive_excess_loss.py, then trains a LoRA adapter on the target model
# using only the top-k% highest-excess-loss tokens per batch.
#
# 
# Sample command:
#   python selective_train_same_tokenizer_ver.py \
#       --task_name word_sorting \
#       --is_train \
#       --k_percent 0.7 \
#       --top_k 250 \
#       --queries_from EXPERT_Mistral-7B-Instruct-v0.3 \
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
from torch.utils.data import DataLoader
from datasets import Dataset
import transformers
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers.integrations import WandbCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from datasets import disable_progress_bar
from transformers.utils import logging
disable_progress_bar()
logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Selective training with excess losses for BBH Tasks")
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
parser.add_argument('--expert_model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
parser.add_argument('--queries_from', type=str, default="EXPERT_Mistral-7B-Instruct-v0.3")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--cut_off', type=int, default=768)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--task_name', type=str, required=True,
                   choices=[
                       'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
                       'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
                       'logical_deduction_five_objects', 'logical_deduction_seven_objects',
                       'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two',
                       'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                       'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding',
                       'temporal_sequences', 'tracking_shuffled_objects_five_objects',
                       'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
                       'web_of_lies', 'word_sorting'
                   ])
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--r', type=int, default=8)
parser.add_argument('--alpha', type=int, default=8)
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--k_percent', type=float, default=0.7)
parser.add_argument('--top_k', type=int, default=250)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Chat template — Mistral only
# ---------------------------------------------------------------------------
MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def apply_chat_template(example, tokenizer):
    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    example["text_prompt"] = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    example["text_chosen"] = tokenizer.apply_chat_template(
        example["chosen"], tokenize=False
    )
    return example

# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------
def load_bbh_data(file_path):
    """Load BBH data from JSON file"""
    print("\n" + "="*60)
    print("DEBUG: LOADING DATA")
    print("="*60)
    print(f"[DEBUG] Loading from: {file_path}")
    print(f"[DEBUG] File extension: {file_path.split('.')[-1]}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")

    with open(file_path, 'r') as f:
        data = json.load(f)

    examples = data.get('examples', [])
    print(f"[DEBUG] Loaded {len(examples)} examples from JSON")

    if examples:
        print(f"[DEBUG] First example keys: {list(examples[0].keys())}")
        print(f"[DEBUG] First input (truncated): {examples[0]['input'][:100]}...")
        print(f"[DEBUG] First target: {examples[0]['target']}")

    return examples


def create_bbh_dataset(train_data, tokenizer, system_prompt):
    """Create dataset from BBH data with proper chat formatting"""
    print("\n" + "="*60)
    print("DEBUG: APPLYING CHAT FORMATTING")
    print("="*60)
    print(f"[DEBUG] System prompt: '{system_prompt}'")
    print(f"[DEBUG] Processing {len(train_data)} samples")

    sft_data = []

    for idx, data in enumerate(train_data[:2]):  # Show first 2 samples
        prompt = data["input"]
        chosen = data["target"]

        if idx < 2:
            print(f"\n[DEBUG] Sample {idx}:")
            print(f"  Raw input: {prompt[:50]}...")
            print(f"  Raw target: {chosen}")

        example = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "chosen": [{"role": "assistant", "content": chosen}]
        }

        formatted_example = apply_chat_template(example, tokenizer)

        if idx < 2:
            print(f"  After template (prompt): {formatted_example['text_prompt'][:100]}...")
            print(f"  After template (chosen): {formatted_example['text_chosen']}")

        sft_data.append({
            "prompt": formatted_example["text_prompt"],
            "chosen": formatted_example["text_chosen"]
        })

    # Process rest without debug
    for data in train_data[2:]:
        prompt = data["input"]
        chosen = data["target"]
        example = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "chosen": [{"role": "assistant", "content": chosen}]
        }
        formatted_example = apply_chat_template(example, tokenizer)
        sft_data.append({
            "prompt": formatted_example["text_prompt"],
            "chosen": formatted_example["text_chosen"]
        })

    return sft_data


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len):
    """Tokenize prompt and chosen response"""
    tokenized_prompt = tokenizer(
        data_point["prompt"],
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False
    )
    tokenized_chosen = tokenizer(
        data_point["chosen"],
        truncation=True,
        max_length=cutoff_len,
        return_tensors=None,
        add_special_tokens=False
    )
    input_ids = tokenized_prompt["input_ids"] + tokenized_chosen["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"] + tokenized_chosen["attention_mask"]
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_chosen["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def load_excess_loss_mapping(excess_loss_path):
    """Load excess loss data and create sample index mapping"""
    print(f"Loading excess loss data from: {excess_loss_path}")

    top_k_path = os.path.join(excess_loss_path, f"top_{args.top_k}_tensors.pt")
    all_data_path = os.path.join(excess_loss_path, "all_excess_losses.pt")

    excess_loss_map = {}

    if os.path.exists(top_k_path):
        print(f"Loading top-k data from: {top_k_path}")
        data = torch.load(top_k_path, map_location='cpu', weights_only=False)

        if 'top_k_tensors' in data:
            for i, item in enumerate(data['top_k_tensors']):
                excess_loss_map[i] = item['excess_losses']
            print(f"Loaded {len(excess_loss_map)} top-k excess loss mappings")
        else:
            raise KeyError("top_k_tensors not found in top-k file")

    elif os.path.exists(all_data_path):
        print(f"Loading all data from: {all_data_path}")
        data = torch.load(all_data_path, map_location='cpu', weights_only=False)

        if 'excess_loss_data' in data:
            sample_idx = 0
            for batch in data['excess_loss_data']:
                batch_size = batch['excess_losses'].shape[0]
                for i in range(batch_size):
                    excess_loss_map[sample_idx] = batch['excess_losses'][i]
                    sample_idx += 1
            print(f"Loaded {len(excess_loss_map)} excess loss mappings from all data")
        else:
            raise KeyError("excess_loss_data not found in all data file")

    else:
        raise FileNotFoundError(f"Neither {top_k_path} nor {all_data_path} found")

    return excess_loss_map

# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class ImprovedSelectiveDataset(torch.utils.data.Dataset):
    def __init__(self, excess_loss_path, top_k):
        print(f"[SafeDataset] Loading Full Context Tensors directly from .pt file...")

        pt_path = os.path.join(excess_loss_path, f"top_{top_k}_tensors.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Could not find {pt_path}")

        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        self.data = []

        list_key = None
        if 'top_k_tensors' in data: list_key = 'top_k_tensors'
        elif 'top_M_tensors' in data: list_key = 'top_M_tensors'
        else:
            raise KeyError(f"Could not find tensor list. Available keys: {list(data.keys())}")

        print(f"[SafeDataset] Found list '{list_key}' with {len(data[list_key])} samples.")

        if len(data[list_key]) > 0:
            first_item = data[list_key][0]
            score_key = None
            possible_keys = ['contrastive_excess_scores', 'excess_losses', 'scores', 'losses']

            for k in possible_keys:
                if k in first_item:
                    score_key = k
                    break

            if score_key is None:
                raise KeyError(f"Could not find scores in item. Available item keys: {list(first_item.keys())}")
            print(f"[SafeDataset] Detected score key: '{score_key}'")
        else:
            score_key = 'excess_losses'

        for item in data[list_key]:
            sample = {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"],
                "excess_losses": item[score_key],
                "has_excess_loss": True,
                "sample_idx": 0
            }

            # Fix alignment
            seq_len = len(sample["input_ids"])
            score_len = len(sample["excess_losses"])
            if score_len == seq_len:
                sample["excess_losses"] = sample["excess_losses"][:-1]

            self.data.append(sample)

        print(f"[SafeDataset] Successfully loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------------------------------------------------------------
# Selective Trainer
# ---------------------------------------------------------------------------
class ImprovedSelectiveTrainer(Trainer):
    def __init__(self, k_percent=0.7, **kwargs):
        self.k_percent = k_percent
        super().__init__(**kwargs)
        self.selected_ratios = []
        self.training_losses = []
        self.excess_losses = []
        self.step_count = 0
        print(f"[ImprovedSelectiveTrainer] Initialized with k_percent={k_percent}")

    def select_top_k_percent_tokens(self, excess_losses, attention_mask, labels):
        """Select top-k% tokens from output positions only"""
        flat_excess = excess_losses.view(-1)
        flat_attention = attention_mask[:, 1:].contiguous().view(-1).bool()

        labels_shifted = labels[:, 1:].contiguous()
        flat_labels = labels_shifted.view(-1)
        output_mask = (flat_labels != -100)

        valid_mask = flat_attention & output_mask

        valid_excess = flat_excess[valid_mask]
        if len(valid_excess) == 0:
            return torch.zeros_like(flat_attention).view(excess_losses.shape), 0

        num_select = max(1, int(len(valid_excess) * self.k_percent))
        _, top_indices = torch.topk(valid_excess, num_select, largest=True)

        selection_mask = torch.zeros_like(flat_attention)
        valid_indices = torch.where(valid_mask)[0]
        selected_indices = valid_indices[top_indices]
        selection_mask[selected_indices] = True
        selection_mask = selection_mask.view(excess_losses.shape)

        selected_count = selection_mask.sum().item()
        total_valid = valid_mask.sum().item()
        actual_ratio = selected_count / total_valid if total_valid > 0 else 0

        return selection_mask, actual_ratio

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute selective loss using excess loss for token selection"""
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        labels = inputs.get('labels')
        excess_losses = inputs.get('excess_losses')
        has_excess_loss = inputs.get('has_excess_loss', torch.ones(input_ids.shape[0], dtype=torch.bool))

        self.step_count += 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if excess_losses is None or not has_excess_loss.any():
            if self.step_count % 50 == 1:
                print(f"  Step {self.step_count}: Using standard loss (no excess loss data)")
            return outputs.loss if not return_outputs else (outputs.loss, outputs)

        # Align excess losses with sequence length
        seq_len = input_ids.shape[1]
        if excess_losses.shape[1] > seq_len - 1:
            excess_losses = excess_losses[:, :seq_len-1]
        elif excess_losses.shape[1] < seq_len - 1:
            pad_len = (seq_len - 1) - excess_losses.shape[1]
            padding = torch.zeros(excess_losses.shape[0], pad_len, device=excess_losses.device, dtype=excess_losses.dtype)
            excess_losses = torch.cat([excess_losses, padding], dim=1)

        valid_mask = attention_mask[:, 1:].bool()
        labels_shifted = labels[:, 1:].contiguous()
        output_mask = (labels_shifted != -100)

        samples_with_excess = has_excess_loss.bool()
        if samples_with_excess.any():
            excess_values = excess_losses[samples_with_excess][valid_mask[samples_with_excess] & output_mask[samples_with_excess]].detach().cpu().numpy()
            if len(excess_values) > 0:
                self.excess_losses.extend(excess_values.tolist())

        selection_mask, actual_ratio = self.select_top_k_percent_tokens(excess_losses, attention_mask, labels)
        self.selected_ratios.append(actual_ratio)

        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        token_losses = token_losses.view(targets.shape)

        if selection_mask.sum() > 0:
            selected_losses = token_losses[selection_mask]
            N = output_mask.sum().item()
            selective_loss = selected_losses.sum() / (N * self.k_percent)
        else:
            selective_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        self.training_losses.append(selective_loss.item())

        if self.step_count % 10 == 0 and wandb.run is not None:
            wandb_metrics = {
                "selective_loss": selective_loss.item(),
                "selection_ratio": actual_ratio,
                "selected_tokens": selection_mask.sum().item(),
                "total_output_tokens": output_mask.sum().item(),
                "k_percent": self.k_percent,
            }
            if len(self.excess_losses) > 0:
                recent_excess = self.excess_losses[-100:]
                wandb_metrics.update({
                    "excess_loss_mean": float(np.mean(recent_excess)),
                    "excess_loss_std": float(np.std(recent_excess)),
                })
            wandb.log(wandb_metrics, commit=False)

        if self.step_count % 50 == 0:
            print(f"  Step {self.step_count}: Loss={selective_loss.item():.3f}, Selection={actual_ratio:.2f}, Tokens={selection_mask.sum().item()}/{output_mask.sum().item()}")

        if return_outputs:
            return selective_loss, outputs
        else:
            return selective_loss

# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
def improved_data_collator(tokenizer, max_length=2048):
    """Data collator that handles excess losses properly"""
    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        excess_losses = [item['excess_losses'] for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        has_excess_loss = torch.tensor([item['has_excess_loss'] for item in batch])
        sample_indices = torch.tensor([item['sample_idx'] for item in batch])

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
                    excess = excess[:max_len-1]

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=lbls.dtype)])

            target_excess_len = max_len - 1
            if len(excess) > target_excess_len:
                excess = excess[:target_excess_len]
            elif len(excess) < target_excess_len:
                excess_pad_len = target_excess_len - len(excess)
                excess = torch.cat([excess, torch.zeros(excess_pad_len, dtype=excess.dtype)])

            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_excess_losses.append(excess)
            padded_labels.append(lbls)

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'excess_losses': torch.stack(padded_excess_losses),
            'labels': torch.stack(padded_labels),
            'has_excess_loss': has_excess_loss,
            'sample_indices': sample_indices,
        }
    return collate_fn

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name, access_token=None):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, token=access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("SELECTIVE TRAINING FOR BBH TASKS")
    print("=" * 60)
    print(f"Model:        {args.model_name}")
    print(f"Expert:       {args.expert_model_name}")
    print(f"Task:         {args.task_name}")
    print(f"K-percent:    {args.k_percent}")
    print(f"Top-K:        {args.top_k}")
    print("=" * 60)

    model_name_short = args.model_name.split('/')[-1]
    expert_short = args.expert_model_name.split('/')[-1]

    # Input: no-align output files from BBH_excess_loss.py
    excess_loss_path = (
        f"./results/excess_losses"
        f"/query_{args.queries_from}_EXPERT_{expert_short}_on_target_{model_name_short}"
        f"/{args.task_name}/"
    )
    print(f"Loading data from: {excess_loss_path}")

    # [1/5] Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    base_model, tokenizer = load_model_and_tokenizer(args.model_name, args.access_token)
    print_trainable_parameters(base_model)

    # Add LoRA
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    user_model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(user_model)

    # [2/5] Create dataset from saved tensors
    print(f"\n[2/5] Creating dataset from saved tensors...")
    dataset = ImprovedSelectiveDataset(
        excess_loss_path=excess_loss_path,
        top_k=args.top_k
    )

    print("="*40)
    print("VERIFYING DATA CONTENT")
    if len(dataset) > 0:
        sample_tensor = dataset[0]['input_ids']
        decoded_text = tokenizer.decode(sample_tensor, skip_special_tokens=False)
        print(f"--- Sample 0 Text Representation ---")
        print(decoded_text)
        print("------------------------------------")
    else:
        print("WARNING: Dataset is empty!")
    print("="*40)

    # Data collator
    data_collator = improved_data_collator(tokenizer, max_length=args.cut_off)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./outputs/',
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
        neftune_noise_alpha=7,
        lr_scheduler_type='linear',
        report_to=['wandb'],
        remove_unused_columns=False
    )

    # [3/5] Train
    print(f"\n[3/5] Starting selective training...")
    trainer = ImprovedSelectiveTrainer(
        model=user_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        k_percent=args.k_percent
    )

    trainer.train()

    # Save model
    save_path = (
        f"./checkpoints/selective"
        f"/{expert_short}_{model_name_short}"
        f"/{args.task_name}/k{args.k_percent}"
    )
    os.makedirs(save_path, exist_ok=True)
    user_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\nModel saved to: {save_path}")
    print("Training completed!")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if args.is_train:
        train(args)
    else:
        print("Inference mode not implemented in this version.")
        print("Please use --is_train flag.")

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()