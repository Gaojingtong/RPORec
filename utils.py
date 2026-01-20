import torch
import numpy as np
from typing import Callable, Dict, Optional
from transformers import AutoTokenizer, EvalPrediction, DataCollatorForLanguageModeling, GenerationConfig
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class MetricUpdater:
    def __init__(self, ks=None):
        self.metric_collection = None

        if ks is None:
            ks = [5, 10, 20, 50]
        self.ks = ks
        self.max_k = max(self.ks)

        # Initialize metric storage
        self._init_metrics()

    def _init_metrics(self):
        self.ndcg_metric = {k: 0. for k in self.ks}
        self.hr_metric = {k: 0. for k in self.ks}
        self.sample_count = 0

    def update(self, logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor):
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Check the input
        valid = _check_valid_input(logits, labels)
        if not valid:
            return

        dcg_k, hit = calculate_metrics(logits, labels, self.max_k)
        # Accumulate NDCG and Hit Rate metrics
        for k in self.ks:
            # NDCG@k
            ndcg_k = dcg_k[:, :k].sum(dim=1)
            # Sum over batch without averaging
            self.ndcg_metric[k] += ndcg_k.sum().item()

            # Hit@k
            hits_k = hit[:, :k].sum(dim=1)
            self.hr_metric[k] += hits_k.sum().item()  # Sum of hits over batch

        self.sample_count += labels.size(0)

    # utils.py (MetricUpdater.compute)
    def compute(self, prefix=""):
        result = {}
        sample_count = self.sample_count
        if sample_count == 0:
            # 无样本，返回 0 或者空 dict，这里返回 0
            for k in self.ks:
                result[prefix + f"ndcg@{k}"] = 0.0
                result[prefix + f"hit_rate@{k}"] = 0.0
            self._init_metrics()
            return result

        for k in self.ndcg_metric:
            result[prefix + f"ndcg@{k}"] = self.ndcg_metric[k] / sample_count
        for k in self.hr_metric:
            result[prefix + f"hit_rate@{k}"] = self.hr_metric[k] / sample_count
        self._init_metrics()
        return result


def _check_valid_input(logits, labels) -> bool:
    # check if empty
    if not logits.numel() or not labels.numel():
        return False

    if logits.size(0) != labels.size(0):
        raise ValueError(
            f"Batch dimension of logits and labels must be the same. Got logits: {logits.size(0)}, labels: {labels.size(0)}")
    # check nan
    if torch.isnan(logits).any():
        raise ValueError("logits contains nan")

    if labels.max().item() >= logits.shape[-1]:
        raise ValueError(
            f"labels contain values greater than the number of classes. Got max label: {labels.max().item()}, num_classes: "
            f"{logits.size(-1)}")

    return True


def calculate_metrics(
        logits: torch.FloatTensor,
        labels: torch.IntTensor,
        cutoff: int,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    """
    Calculate the DCG (Discounted Cumulative Gain) for a batch of predictions and labels.

    Args:
        logits (torch.FloatTensor): The predicted scores for each item, shape: (*, num_items).
        labels (torch.IntTensor): The ground truth labels for each item: (*) or (*, 1)
        cutoff (int): The cutoff value for NDCG calculation.

    Returns:
        torch.FloatTensor: The DCG values for each item in the batch, shape: (*, cutoff).
        torch.BoolTensor: The hit values for each item in the batch, shape: (*, cutoff).

    """
    # labels shape must equal to preds shape except the last dimension
    if len(logits.shape) == len(labels.shape) + 1:
        labels = labels.unsqueeze(-1)
    else:
        assert len(logits.shape) == len(labels.shape), f"{len(logits.shape)} != {len(labels.shape)}"
        assert logits.shape[:-1] == labels.shape[:-1], f"{logits.shape[:-1]} != {labels.shape[:-1]}"
        assert labels.shape[-1] == 1, f"{labels.shape[-1]} != 1"
    _shape = labels.shape[:-1] + (cutoff,)
    labels = labels.expand(_shape)  # (*, cutoff)

    preds = logits.topk(cutoff, dim=-1).indices
    hit = (preds.squeeze(-1) == labels)

    discount = torch.log2(torch.arange(2, cutoff + 2,
                                       dtype=torch.float32,
                                       device=labels.device))
    dcg = (1.0 / discount)  # (cutoff,)
    dcg = torch.where(hit, dcg, 0)  # (*, cutoff)

    return dcg, hit


def get_compute_metrics(metric_updater: MetricUpdater, num_negatives: Optional[int] = None) -> Callable[
        [EvalPrediction, bool], Dict[str, float]]:
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = eval_pred.predictions  # (B, seq, num_items)
        labels = eval_pred.label_ids  # (B, seq)
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        #
        # # exclude batches with no valid samples
        # # valid_batch_index = labels.sum(-1) != -100 * labels.shape[-1]
        # if labels.shape[0] != logits.shape[0]:
        #     device_id = logits.device.index
        #     logits = logits[device_id * labels.shape[0]: (device_id + 1) * labels.shape[0]]
        #     assert labels.shape[0] == logits.shape[0], f"{labels.shape[0]} != {logits.shape[0]}"

        labels = labels.view(-1)

        # logits: (B, num_items), labels: (B,)
        if num_negatives is not None and num_negatives > 0:
            sampled_labels, sampled_logits = _negative_sampling(labels, logits)
            metric_updater.update(logits=sampled_logits, labels=sampled_labels)
        else:
            metric_updater.update(logits=logits, labels=labels)

        result = metric_updater.compute()
        return result

    def _negative_sampling(labels, logits):
        B, num_items = logits.shape
        sampling_prob = torch.ones(
            (B, num_items), dtype=torch.float, device=labels.device)
        sampling_prob[torch.arange(B), labels] = 0  # 将正样本位置的概率设为0
        negative_items = torch.multinomial(
            sampling_prob, num_samples=num_negatives, replacement=False)
        sampled_items = torch.cat([labels.view(-1, 1), negative_items], dim=-1)
        sampled_logits = torch.gather(logits, dim=-1, index=sampled_items)
        sampled_labels = torch.zeros(B, dtype=torch.long, device=labels.device)
        return sampled_labels, sampled_logits

    return compute_metrics


class RecDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, use_columns, **kwargs):
        super().__init__(**kwargs)
        self.use_columns = use_columns
    def torch_call(self, examples):
        batch = {}
        for key in self.use_columns:
            if "attention_mask" in key:
                continue
            elif "input_ids" in key:
                attn_key = key.replace("input_ids", "attention_mask")
                if attn_key not in self.use_columns:
                    raise ValueError("No attention mask related to {}".format(key))
                examples_to_encode = [
                    {
                        "input_ids": e[key],
                        "attention_mask": e[attn_key]
                    } for e in examples]
                inputs = super().torch_call(examples_to_encode)
                batch[key] = inputs["input_ids"]
                batch[attn_key] = inputs["attention_mask"]
            elif key=="seq_labels":
                batch["seq_labels"] = torch.tensor(
                    [e["seq_labels"] for e in examples], dtype=torch.long)
            elif key=="prompt":
                batch["prompt"] = [e["prompt"] for e in examples]
            else:
                raise ValueError("No processing function for column {}".format(key))

        assert len(batch), "No valid input keys found in the examples"
        return batch


def eval_process(model,
                   tokenizer,
                   dataset,
                   batch_size,
                   max_completion_length,
                 item_hs,
                 eval_func
                   ):
    # run eval on main
    # if dist.get_rank() == 0:
    with torch.no_grad():
        device = model.device
        generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if model.training:
            training_flag = True
            model.eval()
        else:
            training_flag = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=RecDataCollator(use_columns=['chat_msg_input_ids', 'chat_msg_attention_mask', 'seq_labels'],
                                       tokenizer=tokenizer, mlm=False),
            shuffle=False,
        )


        pbar = tqdm(
            dataloader, desc="Evaluating...")
        for batch in pbar:
            output = model.generate(
                input_ids=batch["chat_msg_input_ids"].to(device),
                attention_mask=batch["chat_msg_attention_mask"].to(device),
                generation_config=generation_config,
            )

            responses = []
            for i, generated in enumerate(output):
                response = tokenizer.decode(
                    generated[len(batch["chat_msg_input_ids"][i]):], skip_special_tokens=True
                )
                responses.append(response)
            completion_ids = tokenizer(
                responses,
                padding=True,  # Pad to max length in batch
                truncation=False,  # Truncate to model's max length (optional)
                return_tensors="pt",  # Return PyTorch tensors ("tf" for TensorFlow)
            )

            last_hidden = model.forward(
                input_ids=completion_ids["input_ids"].to(device),
                attention_mask=completion_ids["attention_mask"].to(device),
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1][:, -1]
            last_hidden = torch.nn.functional.normalize(last_hidden, dim=1)
            rec_logits = last_hidden @ item_hs.t()
            eval_func.update(rec_logits, batch["seq_labels"].to(device))
        result = eval_func.compute()
        if training_flag:
            model.train()
        return result
        # return 0


# class CustomDataCollator(DataCollatorWithPadding):
#     def __init__(self, tokenizer, fixed_cols, **kwargs):
#         super().__init__(tokenizer, **kwargs)
#         self.fixed_cols = fixed_cols
#
#     def __call__(self, features):
#         # Extract and remove index fields
#         fixed_dict = {}
#         for col in self.fixed_cols:
#             fixed_dict[col] = [f.pop(col) for f in features]
#
#         # Process numerical features with base collator
#         batch = super().__call__(features)
#
#
#         # Add string fields back
#         for col in self.fixed_cols:
#             batch[col] = fixed_dict[col]
#
#         return batch
class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, fixed_cols, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.fixed_cols = fixed_cols

    def __call__(self, features):
        # Extract and remove index fields
        batch = {}
        keys = list(features[0].keys())
        for key in keys:
            if key in self.fixed_cols:
                if key == "seq_labels":
                    # Convert labels to tensor
                    batch[key] = torch.tensor([f[key] for f in features])
                else:
                    batch[key] = [f.pop(key) for f in features]

        if "input_ids" in keys:
            assert "attention_mask" in keys
            prepare_list = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
            batch.update(super().__call__(prepare_list))

        return batch