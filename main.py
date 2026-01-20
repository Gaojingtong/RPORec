from trl import GRPOConfig, GRPOTrainer
import math
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union
import random
from tqdm import tqdm
import datasets
import torch
import torch.utils.data
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from utils import get_compute_metrics, MetricUpdater, eval_process, calculate_metrics, CustomDataCollator
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    is_wandb_available,
)
from accelerate import PartialState
from safetensors.torch import load_file
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from utils import RecDataCollator
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available
from prompters.rec_prompter import UserPrompter, ItemPrompter
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from dataclasses import dataclass, field
from model_cfg.rechead import MulRetriever
from transformers import DataCollatorWithPadding
# import logging
# logger=logging.getLogger(__name__)
if is_peft_available():
    from peft import PeftConfig, get_peft_model


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

class RecTrainer(Trainer):

    def __init__(
            self,
            train_args,
            train_dataset,
            eval_dataset,
            model,
            processing_class,
            **kwargs
    ):

        self.train_args = train_args

        super().__init__(
            model=model,
            args=train_args,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.processing_class.padding_side = "left"
        self.item_hs = None

    def find_tag(self, element):
        cot_msg = []
        ans_msg = []
        format_acc = []
        pre_len = []
        mid_len = []
        after_len = []

        pattern = r'([\s\S]*)<think>([\s\S]*)</think>([\s\S]*)<answer>([\s\S]*)</answer>([\s\S]*)'
        matches = [re.search(pattern, content, re.DOTALL) for content in element["gen_msg"]]
        for i, match in enumerate(matches):
            if not match:
                pre_len.append(0)
                cot_msg.append(element["gen_msg"][i])
                ans_msg.append("")
                mid_len.append(0)
                after_len.append(0)
                format_acc.append(0)
            else:
                pre_len.append(len(match.group(1)))
                cot_msg.append(match.group(2))
                mid_len.append(len(match.group(3)))
                ans_msg.append(match.group(4))
                after_len.append(len(match.group(5)))
                format_acc.append(1)
        return {
            "pre_len": pre_len,
            "cot_msg": cot_msg,
            "mid_len": mid_len,
            "ans_msg": ans_msg,
            "after_len": after_len,
            "format_acc": format_acc
        }

    def _get_sampled_item_hs(self, seq_labels_tensor, device):
        """Helper function to extract sampled item representations for negative sampling."""
        hs_device = self.item_hs.device
        seq_labels_tensor = seq_labels_tensor.to(device)
        if seq_labels_tensor.dim() > 1:
            item_ids = seq_labels_tensor.to(hs_device).long()
            sampled_item_hs = self.item_hs[item_ids]
            return sampled_item_hs, True
        return self.item_hs, False

    def _prepare_label_tensors(self, seq_labels, device):
        """Convert seq_labels into tensors and gather sampled item representations."""
        if not isinstance(seq_labels, torch.Tensor):
            seq_labels_tensor = torch.tensor(seq_labels, device=device)
        else:
            seq_labels_tensor = seq_labels.to(device)

        sampled_item_hs, has_negatives = self._get_sampled_item_hs(seq_labels_tensor, device)
        if has_negatives:
            ce_labels = torch.zeros(seq_labels_tensor.size(0), dtype=torch.long, device=device)
        else:
            ce_labels = seq_labels_tensor.long()
        metric_labels = ce_labels.unsqueeze(-1)
        return sampled_item_hs, ce_labels, metric_labels, has_negatives

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract custom inputs
        device = self.accelerator.device
        model_to_use = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            matches = self.find_tag(inputs)
            matches["chat_msg"] = inputs["chat_msg"]
            prompt_inputs = model_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

            cot_inputs = model_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(device) for k, v in cot_inputs.items()}

            ans_inputs = model_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(device) for k, v in ans_inputs.items()}

            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()
        
        sampled_item_hs, ce_labels, _, _ = self._prepare_label_tensors(inputs["seq_labels"], device)
        ans_logits = model_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                     sampled_item_hs)
        
        ans_loss = F.cross_entropy(ans_logits, ce_labels)

        # use rec_logits in compute_metrics as predictions!!!
        if return_outputs:
            return ans_loss, {"outputs": ans_logits, "loss": ans_loss}
        return ans_loss



class RLTrainer(GRPOTrainer):
    def __init__(
            self,
            **kwargs
    ):

        super().__init__(
            **kwargs
        )
        self.processing_class.padding_side = "left"




@dataclass
class DatasetArguments:
    emb_token: Optional[str] = field(
        default='',
        metadata={"help": "The token to indicate the end of the embedding."},
    )
    emb_end_token: Optional[str] = field(
        default='',
        metadata={"help": "The token to indicate the end of the embedding."},
    )

    dataset_category: Optional[str] = field(
        default='',
        metadata={"help": "The category of the dataset."},
    )
    dataset_window_size: Optional[int] = field(
        default=4,
        metadata={"help": "The window size for user and item input."},
    )
    item_emb_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for embedding item profiles for evaluation."},
    )
    user_input_max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The window size for user and item input."},
    )
    item_input_max_length: Optional[int] = field(
        default=768,
        metadata={"help": "The window size for user and item input."},
    )
    num_negatives: Optional[int] = field(
        default=49,
        metadata={"help": "Number of negative samples per positive sample."},
    )






class AllTrainer:
    def __init__(self, model,
                 frozen_model,
                 processing_class,
                 rl_dataset,
                 rec_head,
                 item_encoder,
                 sft_args,
                 rl_args,
                 dataset_args,
                 sft_metric_updater,
                 eval_func,
                 warmup_path=None,
                 rl=True,):
        self.state = PartialState()
        self.processing_class = processing_class
        self.rl_dataset = rl_dataset
        self.rec_head = rec_head
        self.item_encoder = item_encoder
        self.sft_args = sft_args
        self.rl_args = rl_args
        self.dataset_args = dataset_args
        self.eval_func = eval_func
        self.frozen_model=frozen_model


        train_dataset, eval_dataset = self.prepare_dataset(rl_dataset)
        if warmup_path is None:
            # train
            if self.state.is_main_process:
                print("#### Start SFT Training ####")
            self.rec_head.train()
            # AC_ALLTrainer_v21.py (AllTrainer.__init__)
            self.sft_trainer = RecTrainer(
                train_args=self.sft_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                model=self.rec_head,
                processing_class=self.processing_class,
                data_collator=CustomDataCollator(
                    tokenizer=self.processing_class,
                    fixed_cols=["seq_labels", "chat_msg", "gen_msg"]
                ),
                compute_metrics=get_compute_metrics(sft_metric_updater),
            )
            self.device = self.sft_trainer.accelerator.device

            self.rec_head = self.rec_head.to(self.device)
            self.item_encoder = self.item_encoder.to(self.device)

            self.encode_item_context()
            self.sft_trainer.item_hs = self.item_hs
            if self.state.is_main_process:
                print("#### Training... ####")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.sft_trainer.train()
        elif warmup_path!="frozen":
            # or load from saved path
            if self.state.is_main_process:
                print("#### Load Pretrained RecHead ####")
            state_dict = load_file(warmup_path)
            self.rec_head.load_state_dict(state_dict)
        else:
            if self.state.is_main_process:
                print("#### Use Frozen RecHead ####")

        if rl:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if self.state.is_main_process:
                print("#### Start RL Training ####")
            if self.state.is_main_process:
                print("RL Dataset Len: {}, {}".format(len(train_dataset), len(eval_dataset)))
            self.rl_trainer = RLTrainer(
                model=model,
                args=self.rl_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.processing_class,
                reward_funcs=[self.reward_train, self.reward_eval_loss,
                              self.reward_eval_n5, self.reward_eval_n10, self.reward_eval_n20,
                              self.reward_eval_r5, self.reward_eval_r10, self.reward_eval_r20],
            )
            self.device=self.rl_trainer.accelerator.device
            self.rec_head = self.rec_head.to(self.device)
            self.frozen_model = self.frozen_model.to(self.device)
            if warmup_path is not None:
                self.item_encoder = self.item_encoder.to(self.device)
                self.encode_item_context()
            else:
                self.item_hs = self.item_hs.to(self.device)

            # Prepare for distributed training
            self.rec_head = self.rl_trainer.accelerator.prepare(self.rec_head)
            self.frozen_model = self.rl_trainer.accelerator.prepare(self.frozen_model)
            self.frozen_model.eval()
            self.rec_head_optimizer = torch.optim.AdamW(self.rec_head.parameters(), lr=1e-3)
            self.rec_head_optimizer = self.rl_trainer.accelerator.prepare(self.rec_head_optimizer)
            self.rec_head.eval()

            if self.state.is_main_process:
                print("#### Training... ####")
            self.rl_trainer.train()
        else:
            if self.state.is_main_process:
                print("#### Finished With No RL Training ####")

    def prepare_dataset(self, dataset):
        # Add synchronization barrier at start
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Process item info first to get total number of items
        if self.state.is_main_process:
            dataset['item_info'] = dataset['item_info'].sort("item_id")
            dataset['item_info'] = dataset['item_info'].remove_columns(
                [col for col in dataset['item_info'].column_names
                 if col not in ["item_id", "item_msg"]])

        # Add barrier after item processing
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            if self.state.is_main_process:
                item_dict = dataset['item_info'].to_dict()
            else:
                item_dict = None

            item_list = [item_dict]
            torch.distributed.broadcast_object_list(item_list, src=0)
            dataset['item_info'] = datasets.Dataset.from_dict(item_list[0])

        self.item_dataset = dataset['item_info']
        num_items = len(self.item_dataset)

        def add_negative_samples(examples):
            seq_labels = examples["seq_labels"]
            new_seq_labels = []
            for pos_label in seq_labels:
                negative_candidates = list(range(num_items))
                if pos_label in negative_candidates:
                    negative_candidates.remove(pos_label)
                k = min(self.dataset_args.num_negatives, len(negative_candidates))
                if k == 0:
                    new_seq_labels.append([pos_label])
                    continue
                negatives = random.sample(negative_candidates, k)
                new_seq_labels.append([pos_label] + negatives)
            return {"seq_labels": new_seq_labels}

        for split in ['train', 'valid', 'test']:
            if self.state.is_main_process:
                print(f"Processing for {split}")
                dataset[split] = dataset[split].map(self.find_tag, batched=True)
                dataset[split] = dataset[split].rename_columns({'conv_msg': 'prompt'})
                if split == 'train' and self.dataset_args.num_negatives > 0:
                    print("Negative sampling for training: ", self.dataset_args.num_negatives)
                    dataset[split] = dataset[split].map(add_negative_samples, batched=True)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                if self.state.is_main_process:
                    data_dict = dataset[split].to_dict()
                else:
                    data_dict = None

                data_list = [data_dict]
                torch.distributed.broadcast_object_list(data_list, src=0)
                dataset[split] = datasets.Dataset.from_dict(data_list[0])

        if self.state.is_main_process:
            print(f"Dataset Len: {len(dataset['train'])}, {len(dataset['valid'])}, {len(dataset['test'])}")

        # Final synchronization barrier
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        return dataset['train'], dataset['valid']

    def find_tag(self, element):
        cot_msg = []
        ans_msg = []
        format_acc = []
        pre_len = []
        mid_len = []
        after_len = []

        pattern = r'([\s\S]*)<think>([\s\S]*)</think>([\s\S]*)<answer>([\s\S]*)</answer>([\s\S]*)'
        matches = [re.search(pattern, content, re.DOTALL) for content in element["gen_msg"]]
        for i, match in enumerate(matches):
            if not match:
                pre_len.append(0)
                cot_msg.append(element["gen_msg"][i])
                mid_len.append(0)
                ans_msg.append("")
                after_len.append(0)
                format_acc.append(0)
            else:
                pre_len.append(len(match.group(1)))
                cot_msg.append(match.group(2))
                mid_len.append(len(match.group(3)))
                ans_msg.append(match.group(4))
                after_len.append(len(match.group(5)))
                format_acc.append(1)
        return {
            "pre_len": pre_len,
            "cot_msg": cot_msg,
            "mid_len": mid_len,
            "ans_msg": ans_msg,
            "after_len": after_len,
            "format_acc": format_acc
        }

    def encode_item_context(self):
        device = self.device
        item_hs = None
        if self.state.is_main_process:
            with torch.no_grad():
                np_item = self.item_encoder.encode(self.item_dataset['item_msg'])  # numpy (N, D)
            obj_list = [np_item]
        else:
            obj_list = [None]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(obj_list, src=0)

        # 所有进程从对象列表取出并转为 tensor
        np_item_broadcasted = obj_list[0]
        item_hs = torch.tensor(np_item_broadcasted, device=device, dtype=torch.float32)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.item_hs = item_hs

    def _get_sampled_item_hs(self, seq_labels_tensor, device):
        """Helper function to extract sampled item representations for negative sampling."""
        hs_device = self.item_hs.device
        seq_labels_tensor = seq_labels_tensor.to(device)
        if seq_labels_tensor.dim() > 1:
            item_ids = seq_labels_tensor.to(hs_device).long()
            sampled_item_hs = self.item_hs[item_ids]
            return sampled_item_hs, True
        return self.item_hs, False

    def _prepare_label_tensors(self, seq_labels, device):
        """Convert seq_labels into tensors and gather sampled item representations."""
        if not isinstance(seq_labels, torch.Tensor):
            seq_labels_tensor = torch.tensor(seq_labels, device=device)
        else:
            seq_labels_tensor = seq_labels.to(device)

        sampled_item_hs, has_negatives = self._get_sampled_item_hs(seq_labels_tensor, device)
        if has_negatives:
            ce_labels = torch.zeros(seq_labels_tensor.size(0), dtype=torch.long, device=device)
        else:
            ce_labels = seq_labels_tensor.long()
        metric_labels = ce_labels.unsqueeze(-1)
        return sampled_item_hs, ce_labels, metric_labels, has_negatives

    def reward_train(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):

        if self.state.is_main_process:
            print("global step: ", trainer_state.global_step)

        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            
            sampled_item_hs, ce_labels, metric_labels, has_negatives = self._prepare_label_tensors(
                seq_labels, self.device
            )
            print("Training: {}, has_negatives: {}".format(self.rl_trainer.model.training, has_negatives))

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()


        with torch.no_grad():

            cot_texts = matches["cot_msg"]
            tokenizer = self.processing_class
            if hasattr(self.frozen_model, 'module'):
                model = self.frozen_model.module
            else:
                model = self.frozen_model

            # Build prompts for summarization
            summary_prompts = [self._build_length_prompt(text) for text in cot_texts]

            # Tokenize batch
            tokenized = tokenizer(
                summary_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.dataset_args.user_input_max_length
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # Generate summaries with frozen gradients
            with torch.no_grad():
                model.eval()
                gen_out = model.generate(
                    **tokenized,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.6,
                )

            # Slice out only the generated continuation (remove the prompt part)
            input_ids = tokenized['input_ids']
            generated_strings = []
            summary_pattern = r'([\s\S]*)<answer>([\s\S]*)</answer>([\s\S]*)'
            leng_reward = []
            sim_reward = []
            before_embs = self.item_encoder.encode(cot_texts)
            print("before embs: ", before_embs.shape)
            for i in range(gen_out.size(0)):
                input_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
                gen_ids = gen_out[i][input_len:]
                gen_ori = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                gen_str = re.search(summary_pattern, gen_ori, re.DOTALL)
                if not gen_str:
                    gen_str = " "
                else:
                    gen_str = gen_str.group(2)

                after_emb = self.item_encoder.encode([gen_str])
                emb_sim = self.item_encoder.similarity([before_embs[i]], after_emb).item()
                sim_reward.append(1.0 if emb_sim > 0.5 else 0.0)

                orig_len = max(1, len((cot_texts[i] or "").strip()))
                gen_len = len((gen_str or "").strip())
                ratio = gen_len / orig_len
                leng_reward.append(float(max(0.0, min(1.0, ratio))))

            print("ori_cot: {}\ngenerated summary: {}".format(cot_texts[-1], gen_str))
            leng_reward_tensor = torch.tensor(leng_reward).to(format_reward.device)
            sim_reward_tensor = torch.tensor(sim_reward).to(format_reward.device)

            format_reward = format_reward.squeeze()
            pre_len_list = matches["pre_len"]
            format_acc_list = matches["format_acc"]
            completion_texts = matches["gen_msg"]
            pre_len = torch.tensor(pre_len_list).to(self.device)
            after_len = torch.tensor(matches["after_len"]).to(self.device)
            len_reward = torch.clamp(1 - (pre_len + after_len) / 100.0, min=0.0)
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            dcg_k, _ = calculate_metrics(ans_logits, metric_labels, min(1000, ans_logits.size(-1)))
            ndcg_reward = dcg_k[:, :min(1000, ans_logits.size(-1))].sum(dim=1)

            entropy reward
            tokenizer = self.processing_class
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            if isinstance(completion_ids, torch.Tensor):
                completion_tensor = completion_ids.detach().to(self.device)
                if completion_tensor.dim() == 1:
                    completion_tensor = completion_tensor.unsqueeze(0)
                completion_tensor = completion_tensor.long()
            else:
                max_completion_len = max(
                    (ids.numel() if isinstance(ids, torch.Tensor) else len(ids)) for ids in completion_ids
                ) if completion_ids else 0
                if max_completion_len > 0:
                    completion_tensor = torch.full(
                        (len(completion_ids), max_completion_len),
                        pad_token_id,
                        device=self.device,
                        dtype=torch.long,
                    )
                    for i, ids in enumerate(completion_ids):
                        if isinstance(ids, torch.Tensor):
                            ids_tensor = ids.to(self.device).long()
                        else:
                            ids_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
                        length = ids_tensor.numel()
                        if length > 0:
                            completion_tensor[i, :length] = ids_tensor
                else:
                    completion_tensor = torch.zeros(
                        (len(completion_ids), 0), device=self.device, dtype=torch.long
                    )

            if completion_tensor.size(1) > 0:
                completion_mask = (completion_tensor != pad_token_id).long()
                prompt_encoding = tokenizer(
                    matches["chat_msg"],
                    return_tensors='pt',
                    padding=True,
                    padding_side="left",
                    truncation=True,
                    max_length=getattr(self.rl_trainer, "max_prompt_length",
                                       self.dataset_args.user_input_max_length),
                    add_special_tokens=False,
                )
                prompt_input_ids = prompt_encoding["input_ids"].to(self.device)
                prompt_attention_mask = prompt_encoding["attention_mask"].to(self.device)
                if prompt_input_ids.size(0) != completion_tensor.size(0):
                    prompt_input_ids = prompt_input_ids[:completion_tensor.size(0)]
                    prompt_attention_mask = prompt_attention_mask[:completion_tensor.size(0)]
                input_ids = torch.cat([prompt_input_ids, completion_tensor], dim=1)
                attention_mask = torch.cat([prompt_attention_mask, completion_mask], dim=1)
                _, entropies = self.rl_trainer._get_per_token_logps_and_entropies(
                    self.rl_trainer.model,
                    input_ids,
                    attention_mask,
                    completion_tensor.size(1),
                    batch_size=prompt_input_ids.size(0),
                    compute_entropy=True,
                )
                cot_token_mask = torch.zeros_like(completion_tensor, dtype=torch.bool)
                eos_token_id = tokenizer.eos_token_id
                have_offsets = getattr(tokenizer, "is_fast", False)
                for idx in range(completion_tensor.size(0)):
                    mask_row = completion_mask[idx].bool()
                    valid_len = mask_row.sum().item()
                    if valid_len == 0:
                        continue
                    if format_acc_list[idx] == 0 or not cot_texts[idx]:
                        continue
                    effective_len = valid_len
                    if eos_token_id is not None and valid_len > 0 and completion_tensor[idx][
                        valid_len - 1] == eos_token_id:
                        effective_len = valid_len - 1
                    if effective_len <= 0:
                        continue
                    if have_offsets:
                        comp_text = completion_texts[idx]
                        cot_text = cot_texts[idx]
                        cot_start = pre_len_list[idx] + len("<think>")
                        cot_end = cot_start + len(cot_text)
                        tokenized_completion = tokenizer(
                            comp_text,
                            add_special_tokens=False,
                            return_offsets_mapping=True,
                        )
                        offsets = tokenized_completion["offset_mapping"]
                        ids = torch.tensor(tokenized_completion["input_ids"], device=self.device, dtype=torch.long)
                        ids_len = ids.size(0)
                        compare_len = min(effective_len, ids_len)
                        if compare_len > 0 and not torch.equal(
                                completion_tensor[idx][:compare_len].cpu(),
                                ids[:compare_len].cpu(),
                        ):
                            local_mask_tensor = torch.zeros(effective_len, dtype=torch.bool, device=self.device)
                        else:
                            local_mask = []
                            for start, end in offsets[:effective_len]:
                                if end <= cot_start or start >= cot_end:
                                    local_mask.append(False)
                                else:
                                    local_mask.append(True)
                            local_mask_tensor = torch.tensor(local_mask, device=self.device, dtype=torch.bool)
                            if local_mask_tensor.size(0) < effective_len:
                                pad_size = effective_len - local_mask_tensor.size(0)
                                local_mask_tensor = F.pad(
                                    local_mask_tensor, (0, pad_size), value=False
                                )
                    else:
                        local_mask_tensor = torch.zeros(effective_len, dtype=torch.bool, device=self.device)
                    cot_token_mask[idx, :effective_len] = local_mask_tensor[:effective_len]

                entropy_values = []
                for ent, cot_mask in zip(entropies, cot_token_mask):
                    valid_ent = ent[cot_mask]
                    if valid_ent.numel() == 0:
                        entropy_values.append(0.0)
                        continue
                    ave_entropy = valid_ent.mean()
                    top_k = max(1, math.ceil(valid_ent.numel() * 0.2))
                    top_avg = torch.topk(valid_ent, top_k).values.mean()
                    entropy_values.append((top_avg - ave_entropy).item())
                entropy_reward = torch.tensor(entropy_values, device=self.device)
            else:
                entropy_reward = torch.zeros(ndcg_reward.size(0), device=self.device)

            all_reward = format_reward * (
                        0.1 + len_reward * 0.1 + 0.5 * ndcg_reward + 0.1 * leng_reward_tensor + 0.1 * sim_reward_tensor + 0.1 * entropy_reward)
            # all_reward = format_reward * (
            #         0.1 + len_reward * 0.1 + 0.6 * ndcg_reward + 0.1 * leng_reward_tensor + 0.1 * sim_reward_tensor)
        if self.state.is_main_process:
            print(
                "all reward: {}\nformat reward: {}\nlen reward: {}\nleng reward: {}\nsim reward: {}\nans reward: {}\nentropy reward: {}\n"
                .format(all_reward, format_reward, len_reward, leng_reward, sim_reward, ndcg_reward, entropy_reward))
        # if self.state.is_main_process:
        #     print(
        #         "all reward: {}\nformat reward: {}\nlen reward: {}\nleng reward: {}\nsim reward: {}\nans reward: {}\n"
        #         .format(all_reward, format_reward, len_reward, leng_reward, sim_reward, ndcg_reward))
        print("reward shape: ", all_reward.shape)
        return all_reward.cpu().tolist()


    def reward_eval_loss(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        if self.state.is_main_process:
            print("global step: ", trainer_state.global_step)

        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            sampled_item_hs, ce_labels, _, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()

        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            ans_loss = F.cross_entropy(ans_logits, ce_labels)
        return (torch.ones_like(ce_labels, dtype=torch.float).to(self.device) * ans_loss).cpu().tolist()

    def reward_eval_n10(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            sampled_item_hs, _, label_for_metrics, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()


        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            dcg_k, _ = calculate_metrics(ans_logits, label_for_metrics, min(10, ans_logits.size(-1)))
            reward = dcg_k[:, :min(10, ans_logits.size(-1))].sum(dim=1)
        return reward.cpu().tolist()


    def reward_eval_n20(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            
            sampled_item_hs, _, label_for_metrics, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()


        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            dcg_k, _ = calculate_metrics(ans_logits, label_for_metrics, min(20, ans_logits.size(-1)))
            reward = dcg_k[:, :min(20, ans_logits.size(-1))].sum(dim=1)
        return reward.cpu().tolist()


    def reward_eval_n5(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            
            sampled_item_hs, _, label_for_metrics, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()


        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            dcg_k, _ = calculate_metrics(ans_logits, label_for_metrics, min(5, ans_logits.size(-1)))
            reward = dcg_k[:, :min(5, ans_logits.size(-1))].sum(dim=1)
        return reward.cpu().tolist()

    def reward_eval_r10(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            
            sampled_item_hs, _, label_for_metrics, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()

        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            _, recall_k = calculate_metrics(ans_logits, label_for_metrics, min(10, ans_logits.size(-1)))
            reward = recall_k[:, :min(10, ans_logits.size(-1))].sum(dim=1)
        return reward.cpu().tolist()

    def reward_eval_r20(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            
            sampled_item_hs, _, label_for_metrics, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()

        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            _, recall_k = calculate_metrics(ans_logits, label_for_metrics, min(20, ans_logits.size(-1)))
            reward = recall_k[:, :min(20, ans_logits.size(-1))].sum(dim=1)
        return reward.cpu().tolist()

    def reward_eval_r5(self, prompts, completions, completion_ids, trainer_state, seq_labels, chat_msg, **kwargs):
        rec_head_to_use = self.rec_head
        if hasattr(self.rec_head, 'module'):
            rec_head_to_use = self.rec_head.module

        with torch.no_grad():
            gen_tmp = {"chat_msg": chat_msg, "gen_msg": [content[0]['content'] for content in completions]}
            matches = self.find_tag(gen_tmp)
            matches.update(gen_tmp)
            # print("Example match in reward: ", {key: matches[key][0] for key in matches})
            
            sampled_item_hs, _, label_for_metrics, _ = self._prepare_label_tensors(seq_labels, self.device)

            # Tokenize inputs
            prompt_inputs = rec_head_to_use.retriever1.tokenize(matches["chat_msg"])
            prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
            cot_inputs = rec_head_to_use.retriever2.tokenize(matches["cot_msg"])
            cot_inputs = {k: v.to(self.device) for k, v in cot_inputs.items()}
            ans_inputs = rec_head_to_use.retriever3.tokenize(matches["ans_msg"])
            ans_inputs = {k: v.to(self.device) for k, v in ans_inputs.items()}
            format_reward = torch.tensor(matches["format_acc"]).reshape((-1, 1)).to(self.device)
            format_inputs = torch.cat(
                (1 - format_reward, torch.zeros_like(format_reward), format_reward),
                dim=1).float()

        with torch.no_grad():
            ans_logits = rec_head_to_use(prompt_inputs, cot_inputs, ans_inputs, format_inputs,
                                         sampled_item_hs)
            _, recall_k = calculate_metrics(ans_logits, label_for_metrics, min(5, ans_logits.size(-1)))
            reward = recall_k[:, :min(5, ans_logits.size(-1))].sum(dim=1)
        return reward.cpu().tolist()




    def _build_length_prompt(self, text: str) -> str:
        # Build a concise summarization prompt for the model
        return (
                "You are a highly concise assistant. The following content represents the thinking process for analyzing and predicting the next possible item the user may buy based on his/her interaction history. Please condense the content, extract the key logical reasoning related to this recommendation process, and summarize it into a concise and short summary. You should retain all the core reasoning logic chains and ensure that each reasoning step is described concisely. Please provide the final short summary directly between <answer> and </answer>, without including any intermediate analysis steps, repetitions, titles, or general phrases.\n"
            "Content:\n" + (text or "") + "\n\nShort Summary:<answer>"
        )



if __name__ == '__main__':
    model_name = "Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained('../models/'+model_name)
    frozen_model = AutoModelForCausalLM.from_pretrained('../models/'+model_name)
    model.train()
    tokenizer = AutoTokenizer.from_pretrained("../models/" + model_name, use_fast=True, padding_side='left')
    rl_dataset = datasets.load_from_disk('../data/CDs_and_Vinyl_Qwen3-0.6B')
    rec_head = MulRetriever("../models/static-retrieval-mrl-en-v1")
    item_encoder = SentenceTransformer("../models/static-retrieval-mrl-en-v1")

    # debug command: num_train_epochs, eval_steps
    sft_args = TrainingArguments(
        remove_unused_columns=False,
        label_names=["seq_labels"],
        num_train_epochs=100,
        eval_strategy="steps",  # "no", "steps", "epoch"
        logging_steps=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        eval_steps=20,
        log_level="info",
        logging_first_step=True,
        eval_accumulation_steps=1,
        include_for_metrics=["loss"],
        output_dir="../35acsftoutput",
        logging_dir="../35acsftlog",
        learning_rate=1e-3,
        report_to=["wandb"],
        # report_to="none",
        run_name="35acallrec",
    )

    # debug command: per_device_train_batch_size, per_device_eval_batch_size, num_generations
    rl_args = GRPOConfig(
        reward_weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        top_entropy_quantile=0.2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-6,
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=1,
        num_train_epochs=202.0,
        num_generations=4,
        temperature=1.0,
        # top_p=0.95,
        max_completion_length=1024,
        ddp_find_unused_parameters=False,
        loss_type="grpo",
        beta=0.001,
        output_dir="../35acrloutput",
        logging_dir="../35acrllog",
        run_name="35acallrec",
        save_steps=50,
        save_strategy="steps",
        report_to=["wandb"],
        eval_strategy="steps",
        eval_steps=50,  # Evaluate every 20 steps
        do_eval=True,
        # log_level="info",
        # save_total_limit=1
        eval_on_start=False,
        )
    
    dataset_args = DatasetArguments(
        emb_token='<answer>',
        emb_end_token='</answer>',
        dataset_category="CDs_and_Vinyl",
        dataset_window_size=20,
        user_input_max_length=2048,
        item_input_max_length=768,
        num_negatives=49,
    )
    sft_metric_updater = MetricUpdater(ks=[5, 10, 20])
    metric_updater = [MetricUpdater(ks=[5, 10, 20]) for i in range(4)]
    trainer = AllTrainer(model=model,
                         frozen_model=frozen_model,
                         processing_class=tokenizer,
                         rl_dataset=rl_dataset,
                         rec_head=rec_head,
                         item_encoder=item_encoder,
                         sft_args=sft_args,
                         rl_args=rl_args,
                         dataset_args=dataset_args,
                         sft_metric_updater=sft_metric_updater,
                         eval_func=metric_updater,
                         warmup_path=None,
                         rl=True,)

