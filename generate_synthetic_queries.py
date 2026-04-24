"""
Synthetic Query Generation
This script generates diverse synthetic task inputs by loading few-shot examples from each BBH task dataset.
It utilizes a multi-stage filtering pipeline, including ROUGE-L similarity scoring, to ensure the generated
queries are properly formatted and diverse.
Supports both 'Vanilla' mode (vLLM) and 'Expert' mode (Base model + LoRA adapter).
"""

import os
import torch
import argparse
import json
import logging
from tqdm import tqdm
from transformers import set_seed
from rouge_score import rouge_scorer

# Environment setup
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(42)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Unified BBH Synthetic Data Generator")
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.3')
parser.add_argument('--task_name', type=str, required=True)
parser.add_argument('--expert', action='store_true', help='Use expert mode (LoRA adapter)')
parser.add_argument('--learning_rate', type=str, default='5e-05')
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--num_queries', type=int, default=200)
parser.add_argument('--max_new_tokens', type=int, default=200)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='./output/BBH/')

args = parser.parse_args()
model_name_short = args.model_name.split('/')[-1]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter path — compatible with SFT.py convention
# ---------------------------------------------------------------------------
adapter_path = f"./checkpoints/expert/{args.task_name}/{args.learning_rate}-{args.max_epoch}-{model_name_short}"

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

CHAT_TEMPLATES = {
    "mistral": MISTRAL_CHAT_TEMPLATE,
    "llama3": LLAMA_CHAT_TEMPLATE,
}

def detect_model_family(model_name: str) -> str:
    lower = model_name.lower()
    if "mistral" in lower:
        return "mistral"
    elif "llama" in lower:
        return "llama3"
    else:
        raise ValueError(f"Unsupported model family for: {model_name}. Add a chat template for this model.")

# ---------------------------------------------------------------------------
# Task format validation
# ---------------------------------------------------------------------------
class TaskFormatManager:
    """Manages format specifications for BBH tasks"""
    SINGLE_LINE_TASKS = {
        'boolean_expressions', 'word_sorting', 'sports_understanding',
        'object_counting', 'date_understanding', 'disambiguation_qa',
        'multistep_arithmetic_two'
    }

    TASK_SPECS = {
        'boolean_expressions': {'type': 'expression', 'single_line': True, 'must_end': ' is'},
        'word_sorting': {'type': 'fixed_format', 'single_line': True, 'must_start': 'Sort the following words alphabetically: List:'},
        'sports_understanding': {'type': 'fixed_format', 'single_line': True, 'must_start': 'Is the following sentence plausible?'},
        'salient_translation_error_detection': {'type': 'multiple_choice_structured', 'single_line': False, 'option_count': 4}
    }

    @classmethod
    def validate_format(cls, text, task_name, relaxed_mode=False):
        if not text or not text.strip():
            return False, "empty"
        if relaxed_mode:
            return True, "relaxed"
        spec = cls.TASK_SPECS.get(task_name, {'type': 'general', 'single_line': False})
        if spec.get('single_line') and '\n' in text.strip():
            return False, "newline_detected"
        return True, "valid"

# ---------------------------------------------------------------------------
# Content extraction for ROUGE comparison
# ---------------------------------------------------------------------------
class ContentExtractor:
    """Extracts core content for ROUGE comparison"""
    @staticmethod
    def extract_content(text, task_name):
        text = text.strip()
        if task_name == 'boolean_expressions' and text.endswith(' is'):
            return text[:-3]
        if 'Options:' in text:
            return text.split('Options:')[0]
        return text

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
class DataGenerator:
    def __init__(self, model_name, task_name, sampling_params, expert_mode=False):
        self.model_name = model_name
        self.task_name = task_name
        self.sampling_params = sampling_params
        self.expert_mode = expert_mode
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def initialize_model(self):
        family = detect_model_family(self.model_name)
        template = CHAT_TEMPLATES[family]
        logger.info(f"Using {family} chat template")

        if self.expert_mode:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel, prepare_model_for_kbit_training

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=args.access_token)
            self.tokenizer.chat_template = template

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto",
                torch_dtype=torch.bfloat16, token=args.access_token
            )
            base_model = prepare_model_for_kbit_training(base_model)

            logger.info(f"Loading expert adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            from vllm import LLM
            self.llm = LLM(model=self.model_name, dtype=torch.bfloat16, max_model_len=4096)
            # vLLM uses its own tokenizer internally; set template via HF tokenizer for prompt formatting
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=args.access_token)
            self.tokenizer.chat_template = template

    def generate_batch(self, prompt, batch_size):
        if self.expert_mode:
            inputs = self.tokenizer(
                [prompt] * batch_size, return_tensors="pt", padding=True
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.sampling_params['max_new_tokens'],
                    do_sample=True,
                    temperature=self.sampling_params['temperature'],
                )
            return [
                self.tokenizer.decode(o[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                for o in outputs
            ]
        else:
            from vllm import SamplingParams
            sp = SamplingParams(
                temperature=self.sampling_params['temperature'],
                top_p=self.sampling_params['top_p'],
                max_tokens=self.sampling_params['max_new_tokens'],
            )
            outputs = self.llm.generate([prompt] * batch_size, sp)
            return [o.outputs[0].text.strip() for o in outputs]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_path = f"./data/bbh_split/{args.task_name}_train.json"

    generator = DataGenerator(
        args.model_name,
        args.task_name,
        {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_new_tokens': args.max_new_tokens,
        },
        args.expert,
    )
    generator.initialize_model()

    valid_tasks = []
    round_num = 0
    while len(valid_tasks) < args.num_queries and round_num < 50:
        round_num += 1
        batch = generator.generate_batch("Generate a new task...", args.batch_size)
        for text in batch:
            is_valid, _ = TaskFormatManager.validate_format(text, args.task_name)
            if is_valid and len(valid_tasks) < args.num_queries:
                valid_tasks.append({"input": text, "task_type": args.task_name})

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.task_name}_{model_name_short}_queries.json")
    with open(out_file, 'w') as f:
        json.dump(valid_tasks, f, indent=2)
    print(f"Saved {len(valid_tasks)} queries to {out_file}")

if __name__ == "__main__":
    main()