import os
import sys
sys.path.append("../../")
import ipdb
import json
import pickle
import argparse

from tqdm import tqdm
from random import shuffle
from easydict import EasyDict as edict
from collections import defaultdict, Counter

from utils import set_seed, read_jsonlines, read_json
from processor.processor import EventDetectionProcessor


def parse_sent(text, event_list, nlp):
    doc = nlp(text)
    assert(len(doc.sentences)==1)
    parsed_sentence = doc.sentences[0].words
    assert(len(parsed_sentence)==len(text.split()))

    for event in event_list:
        event_head_word = parsed_sentence[int(event['start'])]
        event['pos'] = event_head_word.upos
        event['lemma'] = text.split()[int(event['start']):int(event['end'])]
        event['lemma'][0] = event_head_word.lemma
        event['lemma'] = " ".join(event['lemma'])


def get_counter_this_sent(example):
    event_counter_this_sent = defaultdict(int)
    last_type = "None"
    for event in example.event_list:
        if event["event_type"] not in ['None', last_type]:
            event_counter_this_sent[event["event_type"]] +=1
        last_type = event["event_type"]
    return event_counter_this_sent


def post_process(filtered_examples, event_type_counter, sample_th):
    post_process_filtered_examples = defaultdict(list)
    for event_type, example_list in filtered_examples.items():
        post_delete_idx_list = list()
        for idx, example in enumerate(example_list):
            event_counter_this_sent = get_counter_this_sent(example)
            flag = True
            for event_type in event_counter_this_sent:
                if event_type_counter[event_type] - event_counter_this_sent[event_type] < sample_th:
                    flag = False
                    break
            if flag:
                post_delete_idx_list.append(idx)
                for event_type in event_counter_this_sent:
                    event_type_counter[event_type] = event_type_counter[event_type] - event_counter_this_sent[event_type]
            else:
                post_process_filtered_examples[event_type].append(example)
    return post_process_filtered_examples


def data_retrieve(args, file_path, sample_th, examples=None):
    set_seed(args.seed)
    filtered_examples = defaultdict(list)
    global_event_type_counter, event_type_counter = defaultdict(int), defaultdict(int)

    if examples is None:
        lines = read_json(file_path) if file_path.endswith("json") else read_jsonlines(file_path)
        processor = EventDetectionProcessor(args=args, tokenizer=None)
        _, examples = processor.create_examples(lines)

    for example in examples:
        for event in filter(lambda x:x["event_type"]!='None', example.event_list):
            global_event_type_counter[event['event_type']] += 1
    event_type_order_list = [a[0] for a in sorted(global_event_type_counter.items(), key=lambda x:x[1])]

    shuffle(examples)
    filtered_example_idx_list, remaining_examples = list(), list()
    for event_type in tqdm(event_type_order_list):
        examples_this_turn = [(idx, example) for idx, example in enumerate(examples) if idx not in filtered_example_idx_list]
        for idx, example in examples_this_turn:
            if event_type_counter[event_type] >= sample_th:
                break
            event_counter_this_sent = get_counter_this_sent(example)

            if event_type in event_counter_this_sent:
                filtered_example_idx_list.append(idx)
                filtered_examples[event_type].append(example)
                for event_type_this_sent, cnt in event_counter_this_sent.items():
                    event_type_counter[event_type_this_sent] += cnt
    remaining_examples = [example for idx, example in enumerate(examples) if idx not in filtered_example_idx_list]

    post_process_filtered_examples = post_process(filtered_examples, event_type_counter, sample_th)
    return post_process_filtered_examples, remaining_examples


def data_vis(examples, parse=True):
    if parse:
        import stanza
        # Note to set 'tokenize_pretokenized' and 'tokenize_no_ssplit' as True.
        # 1. Sentences have been space-separated tokenkenized previously.
        # 2. Each sample is viewed as one and only one sentence.
        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True, tokenize_no_ssplit=True)

    if isinstance(examples, dict):
        output_examples = defaultdict(list)
        for event_type, events in examples.items():
            for event in events:
                output_event = {
                    'sent_idx': event.sent_idx,
                    'sent': " ".join(event.sent),
                    'event_list': [event for event in event.event_list if event['label_id']!=0]
                }
                if parse:
                    parse_sent(output_event['sent'], output_event['event_list'], nlp)
                output_examples[event_type].append(output_event)
    else:
        output_examples = list()
        for event in tqdm(examples):
            output_event = {
                'sent_idx': event.sent_idx,
                'sent': " ".join(event.sent),
                'event_list': [event for event in event.event_list if event['label_id']!=0]
            }
            if parse:
                parse_sent(output_event['sent'], output_event['event_list'], nlp)
            output_examples.append(output_event)

    return output_examples


def data_split_and_save(filtered_examples, output_path, set_type=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if set_type is None:
        set_type = "train"

    train_cnt = defaultdict(int)
    for event_type, sent_list in filtered_examples.items():
        for sent in filtered_examples[event_type]:
            last_type = "None"
            for event in sent.event_list:
                if not (event["event_type"] !="None" and event["event_type"]==last_type):
                    train_cnt[event["event_type"]] += 1
                last_type = event["event_type"]

    pickle.dump(filtered_examples, open(os.path.join(output_path, "{}.pkl".format(set_type)), "wb"))
    with open(os.path.join(output_path, "{}.json".format(set_type)), 'w') as f:
        json.dump(data_vis(filtered_examples), f)
    with open(os.path.join(output_path, "{}_cnt.json".format(set_type)), 'w') as f:
        json.dump(train_cnt, f)


def generate_lemma_counter(examples):
    lemma_dict = defaultdict(list)
    for sent in examples:
        for event in sent["event_list"]:
            lemma_dict[event['event_type']].append(event['lemma'])
    lemma_cnt_dict = {event_type:Counter(lemma_list).most_common() for event_type, lemma_list in lemma_dict.items()}
    return lemma_cnt_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="ACE", type=str, choices=['ACE', 'MAVEN', 'ERE'])
    parser.add_argument("--train_K", default=5, type=int)
    parser.add_argument("--dev_K", default=2, type=int)
    config = parser.parse_args()

    if config.dataset_type=='ACE':
        file_path = "../ACE05_processed/train.json"
        file_test_path = "../ACE05_processed/test.json"
        args = edict(label_dict_path="./label_dict_ACE.json", dataset_type="ACE")
    elif config.dataset_type=='MAVEN':
        file_path = "../MAVEN/train.jsonl"
        file_test_path = "../MAVEN/valid.jsonl"
        args = edict(label_dict_path="./label_dict_MAVEN.json", dataset_type="MAVEN")
    elif config.dataset_type=='ERE':
        file_path = "../ERE/train.jsonl"
        file_test_path = "../ERE/test.jsonl"
        args = edict(label_dict_path="./label_dict_ERE.json", dataset_type="ERE")           
    else:
        raise AssertionError() 

    seed_list = [13, 21, 42, 88, 100, 18, 26, 47, 93, 105]
    for i, seed in enumerate(seed_list):
        args.seed = seed
        output_path = "./fewshot_set/K{}_{}_{}".format(config.train_K, config.dataset_type, i)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        # train set
        filtered_examples, remaining_examples = data_retrieve(args, file_path, sample_th=config.train_K)
        data_split_and_save(filtered_examples, output_path, set_type="train")
        # dev set
        filtered_examples, _ = data_retrieve(args, file_path, sample_th=config.dev_K, examples=remaining_examples)
        data_split_and_save(filtered_examples, output_path, set_type="dev")
