# =============================================================================
# tokenizer_align.py
#
# Takes the align outputs from compute_contrastive_excess_loss.py as input:
#   ./results/excess_losses/query_{queries_from}_EXPERT_{source_short}_on_target_{target_short}/
#     {task_name}/top_{top_k}_output_only_k{k_percent}_binary.pt
#
# Aligns source (Mistral) token-level binary masks to target (Llama-3.1-8B)
# tokenizer space using text-level alignment, then applies per-batch top-k%
# selection over target fractional masks.
#
# Outputs:
#   ./results/tokenizer_aligned/query_{queries_from}_EXPERT_{source_short}_on_target_{target_short}/
#     {task_name}/k{k_percent}_aligned_{target_short}/
#       {task_name}_output_aligned_{target_short}_k{k_percent}.pt
#       {task_name}_alignment_summary_k{k_percent}.json
#
# Only supports Mistral (source) -> Llama-3.1-8B-Instruct (target).
#
# Sample command:
#   python tokenizer_align.py \
#       --source_model mistralai/Mistral-7B-Instruct-v0.3 \
#       --target_model meta-llama/Llama-3.1-8B-Instruct \
#       --queries_from EXPERT_Mistral-7B-Instruct-v0.3 \
#       --task_name word_sorting \
#       --top_k 250 \
#       --align_batch_size 4
# =============================================================================

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from collections import defaultdict
import time

# ---------------------------------------------------------------------------
# Hardcoded chat templates
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
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="BBH Output-Only Tokenizer Alignment")
    parser.add_argument('--source_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--queries_from', type=str, default='EXPERT_Mistral-7B-Instruct-v0.3',
                        help='Query source tag (used for path construction)')
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=250)
    parser.add_argument('--access_token', type=str, default=None)
    parser.add_argument('--k_values', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--single_k', type=float, default=None,
                        help='Process only a single k value')
    parser.add_argument('--continue_on_error', action='store_true')
    parser.add_argument('--align_batch_size', type=int, default=4)
    return parser

# ---------------------------------------------------------------------------
# System prompt — loaded from prompts.json
# ---------------------------------------------------------------------------
def load_prompts(path="./prompts.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_system_prompt(task_name, prompts):
    return prompts.get(task_name, prompts["default"])

# ---------------------------------------------------------------------------
# Per-batch top-k selection
# ---------------------------------------------------------------------------
def batch_select_k_fractionals(fractional_list, k_percent):
    """
    Per-batch top-k% selection across all target tokens in the batch.
    fractional_list: List[np.ndarray], one per sample.
    Returns: List[np.ndarray of int32], 1 for selected tokens else 0.
    """
    offsets = []
    flat = []
    total = 0
    for frac in fractional_list:
        n = len(frac)
        offsets.append((total, total + n))
        flat.append(frac.astype(np.float32))
        total += n
    if total == 0:
        return [np.zeros_like(frac, dtype=np.int32) for frac in fractional_list]

    flat = np.concatenate(flat, axis=0)
    num_select = max(1, int(total * k_percent))
    if num_select >= total:
        selected = np.ones(total, dtype=np.int32)
    else:
        top_idx = np.argpartition(-flat, num_select - 1)[:num_select]
        selected = np.zeros(total, dtype=np.int32)
        selected[top_idx] = 1

    return [selected[s:e].copy() for s, e in offsets]

# ---------------------------------------------------------------------------
# Chat rendering helpers
# ---------------------------------------------------------------------------
def render_input_only(tokenizer, system_prompt, user_text):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids

def render_full_with_answer(tokenizer, system_prompt, user_text, answer_text):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer_text},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids

def extract_assistant_span(tokenizer, system_prompt, user_text, answer_text):
    """Return (assistant_token_ids, per_token_texts, assistant_decoded_text)."""
    _, input_only_ids = render_input_only(tokenizer, system_prompt, user_text)
    _, full_ids = render_full_with_answer(tokenizer, system_prompt, user_text, answer_text)
    ass_ids = full_ids[len(input_only_ids):]
    tok_texts = []
    for tid in ass_ids:
        try:
            tok_texts.append(tokenizer.decode([tid], skip_special_tokens=True))
        except Exception:
            tok_texts.append(f"<UNK_{tid}>")
    ass_text = tokenizer.decode(ass_ids, skip_special_tokens=True)
    return ass_ids, tok_texts, ass_text

# ---------------------------------------------------------------------------
# Tokenizer aligner
# ---------------------------------------------------------------------------
class TokenizerAligner:
    """Aligns source (Mistral) binary masks to target (Llama3) token space."""

    def __init__(self, source_tokenizer, target_tokenizer):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def _normalize_text(self, text):
        return text.strip().replace(' ', '').replace('\n', '').replace('\t', '')

    def _align_token_sequences(self, source_tokens, target_tokens, source_text, target_text):
        source_norm = self._normalize_text(source_text)
        target_norm = self._normalize_text(target_text)
        text_mismatch = {
            'has_mismatch': source_norm != target_norm,
            'source_len': len(source_text),
            'target_len': len(target_text),
            'major_difference': abs(len(source_norm) - len(target_norm)) > 5,
        }
        alignments = []
        source_ptr = 0
        target_ptr = 0
        while source_ptr < len(source_tokens) and target_ptr < len(target_tokens):
            source_segment = ""
            source_start = source_ptr
            while source_ptr < len(source_tokens):
                source_segment += source_tokens[source_ptr]
                source_ptr += 1
                target_segment = ""
                target_start = target_ptr
                temp_target_ptr = target_ptr
                while temp_target_ptr < len(target_tokens):
                    target_segment += target_tokens[temp_target_ptr]
                    temp_target_ptr += 1
                    if (self._normalize_text(source_segment) == self._normalize_text(target_segment)
                            and self._normalize_text(source_segment)):
                        alignments.append(((source_start, source_ptr), (target_start, temp_target_ptr)))
                        target_ptr = temp_target_ptr
                        break
                if (self._normalize_text(source_segment) == self._normalize_text(
                        "".join(target_tokens[target_start:target_ptr]))
                        and self._normalize_text(source_segment)):
                    break
        return alignments, text_mismatch

    def _apply_alignment_rules(self, source_masks, alignments, source_tokens, target_tokens):
        max_target_idx = max((te for _, (_, te) in alignments), default=0)
        if max_target_idx == 0:
            return np.zeros(len(target_tokens), dtype=np.float32), defaultdict(int)

        target_fractional = np.zeros(max_target_idx, dtype=np.float32)
        stats = defaultdict(int)

        for (source_start, source_end), (target_start, target_end) in alignments:
            source_len = source_end - source_start
            target_len = target_end - target_start
            if source_end <= len(source_masks):
                source_values = source_masks[source_start:source_end]
            else:
                available = max(0, len(source_masks) - source_start)
                if available > 0:
                    source_values = source_masks[source_start:source_start + available]
                else:
                    continue
            if len(source_values) == 0:
                stats['exceptions'] += 1
                continue
            if source_len == 1 and target_len == 1:
                target_fractional[target_start] = source_values[0]
                stats['one_to_one'] += 1
            elif source_len == 1 and target_len > 1:
                target_fractional[target_start:target_end] = source_values[0]
                stats['one_to_many'] += 1
            elif source_len > 1 and target_len == 1:
                target_fractional[target_start] = np.mean(source_values)
                stats['many_to_one'] += 1
            elif source_len > 1 and target_len > 1:
                target_fractional[target_start:target_end] = np.mean(source_values)
                stats['many_to_many'] += 1
            else:
                stats['exceptions'] += 1

        return target_fractional, stats

    def align_output_sample(self, input_text, output_text, source_binary_mask,
                            k_percent, sample_idx=None, system_prompt=""):
        src_ass_ids, src_ass_toktxt, src_ass_text = extract_assistant_span(
            self.source_tokenizer, system_prompt, input_text, output_text
        )
        tgt_ass_ids, tgt_ass_toktxt, tgt_ass_text = extract_assistant_span(
            self.target_tokenizer, system_prompt, input_text, output_text
        )
        if len(src_ass_ids) == 0 or len(tgt_ass_ids) == 0:
            return None

        src_mask = np.array(source_binary_mask, dtype=np.float32)
        if len(src_mask) != len(src_ass_ids):
            old_idx = np.linspace(0, max(0, len(src_mask) - 1), num=len(src_ass_ids))
            src_mask = src_mask[np.round(old_idx).astype(int)]

        alignments, text_mismatch = self._align_token_sequences(
            src_ass_toktxt, tgt_ass_toktxt, src_ass_text, tgt_ass_text
        )

        if len(alignments) == 0:
            tgt_fractional = np.zeros(len(tgt_ass_ids), dtype=np.float32)
            m = min(len(src_mask), len(tgt_fractional))
            tgt_fractional[:m] = src_mask[:m]
            alignment_stats = {'fallback': 1}
        else:
            tgt_fractional, stats = self._apply_alignment_rules(
                src_mask, alignments, src_ass_toktxt, tgt_ass_toktxt
            )
            alignment_stats = dict(stats)
            if len(tgt_fractional) != len(tgt_ass_ids):
                if len(tgt_fractional) > len(tgt_ass_ids):
                    tgt_fractional = tgt_fractional[:len(tgt_ass_ids)]
                else:
                    tgt_fractional = np.concatenate([
                        tgt_fractional,
                        np.zeros(len(tgt_ass_ids) - len(tgt_fractional), dtype=np.float32)
                    ])

        source_selected = float(np.sum(src_mask))
        source_ratio = source_selected / len(src_mask) if len(src_mask) > 0 else 0.0

        return {
            'sample_idx': sample_idx,
            'source_output_tokens': src_ass_ids,
            'source_binary_mask': src_mask.tolist(),
            'target_output_tokens': tgt_ass_ids,
            'target_fractional_mask': tgt_fractional.tolist(),
            'alignments': alignments,
            'alignment_stats': alignment_stats,
            'text_mismatch': text_mismatch,
            'selection_stats_prelim': {
                'source_selected': int(source_selected),
                'source_ratio': float(source_ratio),
                'k_percent': k_percent,
            },
        }

# ---------------------------------------------------------------------------
# Main alignment processor
# ---------------------------------------------------------------------------
class BBHTokenizerAlignment:

    def __init__(self, args):
        self.args = args
        self.source_tokenizer = None
        self.target_tokenizer = None
        self.prompts = load_prompts("./prompts.json")

    def _load_tokenizers(self):
        print("Loading tokenizers...")
        self.source_tokenizer = AutoTokenizer.from_pretrained(
            self.args.source_model, token=self.args.access_token
        )
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_model, token=self.args.access_token
        )
        if self.source_tokenizer.pad_token is None:
            self.source_tokenizer.pad_token = self.source_tokenizer.eos_token
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        # Hardcoded: source = Mistral, target = Llama3
        self.source_tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        self.target_tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

        print(f"Source tokenizer: {len(self.source_tokenizer):,} tokens")
        print(f"Target tokenizer: {len(self.target_tokenizer):,} tokens")

    def _find_input_file(self, k_percent):
        source_short = self.args.source_model.split('/')[-1]
        target_short = self.args.target_model.split('/')[-1]
        path = (
            f"./results/excess_losses"
            f"/query_{self.args.queries_from}_EXPERT_{source_short}_on_target_{target_short}"
            f"/{self.args.task_name}"
            f"/top_{self.args.top_k}_output_only_k{k_percent}_binary.pt"
        )
        return path if os.path.exists(path) else None

    def _create_output_dir(self, k_percent):
        source_short = self.args.source_model.split('/')[-1]
        target_short = self.args.target_model.split('/')[-1]
        output_dir = (
            f"./results/tokenizer_aligned"
            f"/query_{self.args.queries_from}_EXPERT_{source_short}_on_target_{target_short}"
            f"/{self.args.task_name}"
            f"/k{k_percent}_aligned_{target_short}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def process_k_value(self, k_percent):
        print(f"\nProcessing k={k_percent}...")

        input_path = self._find_input_file(k_percent)
        if not input_path:
            raise FileNotFoundError(f"Input file not found for k={k_percent}")
        print(f"  Input: {os.path.basename(input_path)}")

        input_data = torch.load(input_path, map_location='cpu', weights_only=False)
        output_samples = input_data['k_binary_data']
        print(f"  Samples: {len(output_samples)}")

        aligner = TokenizerAligner(self.source_tokenizer, self.target_tokenizer)
        system_prompt = get_system_prompt(self.args.task_name, self.prompts)

        batch_buf = []
        batch_frac_list = []
        aligned_results = []
        failed_count = 0
        group_id = 0

        for i, sample in enumerate(tqdm(output_samples, desc=f"Aligning k={k_percent}")):
            try:
                result = aligner.align_output_sample(
                    input_text=sample['input_text'],
                    output_text=sample['output_text'],
                    source_binary_mask=sample['binary_mask'],
                    k_percent=k_percent,
                    sample_idx=i,
                    system_prompt=system_prompt,
                )
                if result is None:
                    failed_count += 1
                    if not self.args.continue_on_error:
                        raise RuntimeError("Alignment returned None")
                    continue

                result.update({
                    'original_sample_idx': sample['sample_idx'],
                    'input_text': sample['input_text'],
                    'output_text': sample['output_text'],
                    'expected_output': sample.get('expected_output', None),
                })

                batch_buf.append(result)
                batch_frac_list.append(np.asarray(result['target_fractional_mask'], dtype=np.float32))

                if len(batch_buf) == self.args.align_batch_size:
                    batch_binaries = batch_select_k_fractionals(batch_frac_list, k_percent)
                    for r, bin_mask in zip(batch_buf, batch_binaries):
                        bin_mask = np.asarray(bin_mask, dtype=np.int32)
                        r['target_binary_mask'] = bin_mask.tolist()
                        r['align_group_id'] = int(group_id)
                        tgt_sel = int(bin_mask.sum())
                        tgt_len = int(bin_mask.size)
                        prelim = r.pop('selection_stats_prelim', {})
                        r['selection_stats'] = {
                            **prelim,
                            'target_selected': tgt_sel,
                            'target_ratio': (tgt_sel / tgt_len) if tgt_len > 0 else 0.0,
                        }
                        aligned_results.append(r)
                    batch_buf.clear()
                    batch_frac_list.clear()
                    group_id += 1

            except Exception as e:
                print(f"    Error processing sample {i}: {e}")
                failed_count += 1
                if not self.args.continue_on_error:
                    raise

        # Flush remaining partial batch
        if batch_buf:
            batch_binaries = batch_select_k_fractionals(batch_frac_list, k_percent)
            for r, bin_mask in zip(batch_buf, batch_binaries):
                bin_mask = np.asarray(bin_mask, dtype=np.int32)
                r['target_binary_mask'] = bin_mask.tolist()
                r['align_group_id'] = int(group_id)
                tgt_sel = int(bin_mask.sum())
                tgt_len = int(bin_mask.size)
                prelim = r.pop('selection_stats_prelim', {})
                r['selection_stats'] = {
                    **prelim,
                    'target_selected': tgt_sel,
                    'target_ratio': (tgt_sel / tgt_len) if tgt_len > 0 else 0.0,
                }
                aligned_results.append(r)

        print(f"  Processed: {len(aligned_results)}/{len(output_samples)} samples")
        if failed_count > 0:
            print(f"  Failed: {failed_count}")

        total_source_tokens = sum(len(r['source_output_tokens']) for r in aligned_results)
        total_target_tokens = sum(len(r['target_output_tokens']) for r in aligned_results)
        total_source_selected = sum(r['selection_stats'].get('source_selected', 0) for r in aligned_results)
        total_target_selected = sum(r['selection_stats'].get('target_selected', 0) for r in aligned_results)

        overall_stats = {
            'total_source_tokens': total_source_tokens,
            'total_target_tokens': total_target_tokens,
            'total_source_selected': total_source_selected,
            'total_target_selected': total_target_selected,
            'source_selection_ratio': (total_source_selected / total_source_tokens) if total_source_tokens > 0 else 0.0,
            'target_selection_ratio': (total_target_selected / total_target_tokens) if total_target_tokens > 0 else 0.0,
        }

        alignment_stats = defaultdict(int)
        for r in aligned_results:
            for key, value in r.get('alignment_stats', {}).items():
                alignment_stats[key] += value

        return {
            'task_name': self.args.task_name,
            'source_model': self.args.source_model,
            'target_model': self.args.target_model,
            'k_percent': k_percent,
            'top_k': self.args.top_k,
            'aligned_output_samples': aligned_results,
            'overall_stats': overall_stats,
            'alignment_statistics': dict(alignment_stats),
            'quality_metrics': {
                'failed_samples': failed_count,
                'success_rate': (len(aligned_results) / len(output_samples) * 100.0) if output_samples else 0.0,
            },
            'original_input_data': {
                'input_path': input_path,
                'config': input_data.get('config', {}),
                'binary_info': input_data.get('binary_info', {}),
            },
        }

    def save_results(self, output_data, k_percent, runtime_sec=None):
        output_dir = self._create_output_dir(k_percent)
        target_short = self.args.target_model.split('/')[-1]

        output_file = f'{self.args.task_name}_output_aligned_{target_short}_k{k_percent}.pt'
        output_path = os.path.join(output_dir, output_file)
        torch.save(output_data, output_path)

        summary = {
            'task_name': self.args.task_name,
            'models': f"{self.args.source_model.split('/')[-1]} → {target_short}",
            'k_percent': k_percent,
            'overall_stats': output_data['overall_stats'],
            'quality_metrics': output_data['quality_metrics'],
            'alignment_patterns': output_data['alignment_statistics'],
            'files': {
                'main': output_file,
                'summary': f'{self.args.task_name}_alignment_summary_k{k_percent}.json',
            },
        }
        if runtime_sec is not None:
            summary['runtime_sec'] = runtime_sec
            output_data['runtime_sec'] = runtime_sec

        summary_path = os.path.join(output_dir, summary['files']['summary'])
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return output_path, summary_path

    def run(self):
        print(f"{'='*80}")
        print(f"BBH TOKENIZER ALIGNMENT — Mistral → Llama-3.1-8B")
        print(f"{'='*80}")
        print(f"Task:   {self.args.task_name}")
        print(f"Source: {self.args.source_model}")
        print(f"Target: {self.args.target_model}")

        self._load_tokenizers()
        k_values = [self.args.single_k] if self.args.single_k is not None else self.args.k_values
        print(f"K values: {k_values}")

        results = {}
        successful = 0
        failed = 0

        for k_percent in k_values:
            try:
                start_time = time.time()
                output_data = self.process_k_value(k_percent)
                runtime_sec = time.time() - start_time
                output_path, summary_path = self.save_results(output_data, k_percent, runtime_sec)
                stats = output_data['overall_stats']
                quality = output_data['quality_metrics']
                results[k_percent] = {
                    'status': 'success',
                    'output_path': output_path,
                    'summary_path': summary_path,
                    'source_ratio': stats['source_selection_ratio'],
                    'target_ratio': stats['target_selection_ratio'],
                    'success_rate': quality['success_rate'],
                    'runtime_sec': runtime_sec,
                }
                successful += 1
                print(
                    f"  ✓ k={k_percent}: "
                    f"target ratio={stats['target_selection_ratio']:.3f}, "
                    f"success={quality['success_rate']:.1f}%, "
                    f"time={runtime_sec:.2f}s"
                )
            except Exception as e:
                print(f"  ✗ k={k_percent}: {str(e)}")
                results[k_percent] = {'status': 'failed', 'error': str(e)}
                failed += 1
                if not self.args.continue_on_error:
                    break

        print(f"\n{'='*80}")
        print(f"ALIGNMENT SUMMARY — {self.args.task_name}")
        print(f"Successful: {successful}/{len(k_values)}  |  Failed: {failed}")
        for k_percent, res in results.items():
            if res['status'] == 'success':
                print(f"   k={k_percent}: target ratio={res['target_ratio']:.3f}, success={res['success_rate']:.1f}%")
            else:
                print(f"   k={k_percent}: {res['error']}")
        print(f"{'='*80}")
        return results

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    processor = BBHTokenizerAlignment(args)
    results = processor.run()
    return any(r['status'] == 'success' for r in results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)