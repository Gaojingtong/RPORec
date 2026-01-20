from prompters.rec_prompter import UserPrompter_v1 as UserPrompter
import datasets
from datasets import concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


@dataclass
class DatasetArguments():
    emb_token: Optional[str] = field(
        default='',
        metadata={"help": "The token to indicate the end of the embedding."},
    )
    emb_end_token: Optional[str] = field(
        default='',
        metadata={"help": "The token to indicate the end of the embedding."},
    )
    user_input_max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The max length of user input."},
    )
    max_completion_length: Optional[int] = field(
        default=768,
        metadata={"help": "The max length of item input."},
    )
    dataset_category: Optional[str] = field(
        default='',
        metadata={"help": "The category of the dataset."},
    )
    dataset_window_size: Optional[int] = field(
        default=4,
        metadata={"help": "The window size for user and item input."},
    )

def prepare_dataset(dataset, args, tokenizer, model, generation_config):

    user_prompter = UserPrompter(dset=dataset,
                                 category=args.dataset_category,
                                 window_size=args.dataset_window_size,
                                 tokenizer=tokenizer,
                                 emb_token=args.emb_token,
                                 emb_end_token=args.emb_end_token
                                 )
    print("Prompters initialized, start converting datasets...", end='')

    for split in ['train', 'valid', 'test']:
        # user description

        dataset[split] = dataset[split].rename_columns({'history_item_id': 'seq_input_ids',
                                                        'item_id': 'seq_labels'})
        dataset[split] = user_prompter.convert_dataset(
            dset=dataset[split])
        dataset[split] = dataset[split].remove_columns(
            [col for col in dataset[split].column_names
             if col not in ["chat_msg", "seq_labels", "conv_msg"]])
    # dataset['train'] = concatenate_datasets([dataset['train']]*10)

    dataset['item_info'] = dataset['item_info'].remove_columns(
        [col for col in dataset['item_info'].column_names
         if col not in ["item_id", "item_msg"]])
    dataset['item_info']=dataset['item_info'].sort("item_id")
    for split in ['train', 'valid', 'test']:
        dataset[split] = dataset[split].map(
            generate_batch,
            batched=True,
            batch_size=12,  # Adjust based on GPU memory
            fn_kwargs={
                'model': model,
                'tokenizer': tokenizer,
                'generation_config': generation_config,
                'user_input_max_length': args.user_input_max_length
            }
        )


    return dataset


def generate_batch(examples, model, tokenizer, generation_config, user_input_max_length):
    inputs = tokenizer(
        examples['chat_msg'],
        add_special_tokens=False,
        truncation=True,
        padding='longest',
        padding_side='left',
        max_length=user_input_max_length,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt"
    ).to(model.device)

    # Generate outputs
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
        )

    # Decode skipping special tokens
    decoded = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return {"gen_msg": [text.strip() for text in decoded]}


if __name__ == '__main__':
    # model_name = "Qwen3-14B-AWQ"
    model_name = "Qwen3-0.6B"
    # load and process dataset
    dataset = datasets.load_from_disk('../data/CDs_and_Vinyl_0_2022-10-2023-10')
    tokenizer = AutoTokenizer.from_pretrained("../models/" + model_name, use_fast=True, padding_side='left')

    args = DatasetArguments(
        emb_token='<answer>',
        emb_end_token='</answer>',
        dataset_category="CDs_and_Vinyl",
        dataset_window_size=20,
        user_input_max_length=2048,
        max_completion_length=768,
    )


    model = AutoModelForCausalLM.from_pretrained(
        '../models/' + model_name,
        device_map="auto",
        trust_remote_code=True
    ).eval().to('cuda')

    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Important for batch padding
    )

    dataset = prepare_dataset(dataset=dataset,
                              args=args,
                              tokenizer=tokenizer,
                              model=model,
                              generation_config=generation_config,)
    data_name = f"{model_name}"
    dataset.save_to_disk("../data/CDs_and_Vinyl_"+data_name)




