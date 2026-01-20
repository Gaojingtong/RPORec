from datetime import datetime
from typing import Optional

import datasets

# from prompters.abstract_prompter import AbstractPrompter
from prompters.prompts import obtain_prompts

DESCRIPTION_MAX_LEN = 100




def get_item_info_str(sequence, item_dset, just_title=False):
    if just_title:
        item_info_str = sequence['item_title'] if "item_title" in sequence else sequence['title']
        return item_info_str
    if "title" in sequence:  # this is item_info split
        item_title = sequence['title']
        item_info = sequence
    else:  # this is trn/test/val split
        item_id = sequence['seq_labels']
        item_title = sequence['item_title']
        item_info = item_dset[item_id - 1]
        item_title_2 = item_info['title']
        assert item_title == item_title_2, f"item_title: {item_title}, item_title_2: {item_title_2}"

    average_rating = item_info['average_rating']
    num_buyers = item_info['rating_number']
    description = item_info['description']

    description = "" if len(description) == 0 else ' '.join(description[::-1])
    des = description.split()
    len_des = len(des)
    if len_des > DESCRIPTION_MAX_LEN:
        description = ' '.join(des[:DESCRIPTION_MAX_LEN]) + '...'

    item_info_str = (f"Title: {item_title}\n"
                     f"User Rating: {average_rating}\n"
                     f"Number of Buyers: {num_buyers}\n"
                     f"Description: {description}")
    # item_info_str = item_title
    return item_info_str


def format_timedelta(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    minutes += seconds / 60
    if days > 0:
        # turn hours, minutes into float hours
        hours += minutes / 60
        _str = f"{days}d {hours:.1f}h"
    else:
        if hours > 0:
            _str = f"{hours}h {minutes:.1f}min"
        else:
            _str = f"{minutes:.1f}min"
    _str += ' ago'
    return _str


def timestamp2str(example):
    history_timestamp = example['history_timestamp']
    timestamp = example['timestamp']
    history_timestamp = [datetime.fromtimestamp(
        t/1000) for t in history_timestamp]
    timestamp = datetime.fromtimestamp(timestamp/1000)
    delta_times = [timestamp - t for t in history_timestamp]
    human_readable = [format_timedelta(delta) for delta in delta_times]
    return human_readable









def get_items_str_v1(sequence, item_dset, win_size=10):
    strs = []
    dset_win_size = len(sequence['seq_input_ids'])
    _time_delta_strs = timestamp2str(sequence)
    # assert win_size <= dset_win_size, f"win_size: {win_size}, dset_win_size: {dset_win_size}"
    start = 0 if dset_win_size <= win_size else dset_win_size - win_size
    for i in range(start, dset_win_size):
        item_id = sequence['seq_input_ids'][i]
        item_info = item_dset[item_id]
        strs.append(item_info['item_msg'])
    history_items = '\n'.join(strs)
    return history_items


class ItemPrompter():

    def __init__(self,
                 category,
                 tokenizer,
                 dset=None,
                 emb_token='',
                 emb_end_token='',
                 ):
        self.tokenizer = tokenizer
        self.dset = dset
        self.prompt = obtain_prompts(category)["item_prompt"].format(emb_token=emb_token, emb_end_token=emb_end_token)

    def convert_dataset(self,
                        dset: datasets.Dataset = None,
                        ):
        return dset




class UserPrompter_v1():

    def __init__(self,
                 category,
                 tokenizer,
                 dset=None,
                 window_size=10,
                 emb_token='',
                 emb_end_token='',
                 ):
        self.dset = dset
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.prompt = obtain_prompts(category)['user_prompt'].format(emb_token=emb_token, emb_end_token=emb_end_token)

    def to_chat_example(self, sequence):
        history_items = get_items_str_v1(
            sequence, self.dset['item_info'], self.window_size)
        sequence["conv_msg"] = [{"role": "assistant",
                                 "content": "You are a helpful assistant. You can help me by answering my questions."},
                                {"role": "user", "content": self.prompt + '\n' + history_items}, ]
        return sequence

    def convert_dataset(self,
                        dset: datasets.Dataset = None,
                        ):
        # check window size
        seqs = dset['seq_input_ids']
        seq_max_len = max([len(seq) for seq in seqs])
        assert self.window_size <= seq_max_len, f"Invalid window size: {self.window_size}, seq_max_len: {seq_max_len}"
        # add conv_msg
        dset = dset.map(self.to_chat_example,
                        desc='Converting to chat examples',
                        batched=False,
                        keep_in_memory=True
                        )

        dset = dset.map(self.formatting_func, batched=True,
                        fn_kwargs={
                            'key': 'conv_msg',
                            'tokenizer': self.tokenizer,
                        })

        return dset

    def formatting_func(self, element, key, tokenizer):
        conversations = []
        for i in range(len(element[key])):
            conversation = tokenizer.apply_chat_template(element[key][i],
                                                         tokenize=False,
                                                         add_generation_prompt=True, )

            conversations.append(conversation)
        result = {'chat_msg': conversations}
        return result