import sys
sys.path.append("../")
sys.path.append("../../")
import os
import json
import pickle
import argparse

from tqdm import tqdm
from random import shuffle, uniform
from easydict import EasyDict as edict
from collections import defaultdict, OrderedDict
from utils import read_json, read_jsonlines, set_seed
from processor.processor import EventDetectionProcessor
from k_shot.construct_fewshot_dataset import data_split_and_save, get_counter_this_sent, post_process


def generate_data(args, file_path, target_event_type_num=None):
    lines = read_json(file_path) if file_path.endswith("json") else read_jsonlines(file_path)
    processor = EventDetectionProcessor(args=args, tokenizer=None)
    _, examples = processor.create_examples(lines)
    
    if target_event_type_num is not None:
        global_event_type_counter = defaultdict(int)
        for example in examples:
            for event in filter(lambda x:x["event_type"]!='None', example.event_list):
                global_event_type_counter[event['event_type']] += 1
        event_type_order_list = [a[0] for a in sorted(global_event_type_counter.items(), key=lambda x:x[1])]
        target_event_type_list, source_event_type_list = event_type_order_list[:target_event_type_num], event_type_order_list[target_event_type_num:]

        source_label_dict, target_label_dict = OrderedDict(), OrderedDict()
        source_label_dict["None"] = 0
        target_label_dict["None"] = 0
        for event_type in source_event_type_list:
            source_label_dict[event_type] = len(source_label_dict)
        for event_type in target_event_type_list:
            target_label_dict[event_type] = len(target_label_dict)

        with open(args.source_label_path, 'w') as f:
            json.dump(source_label_dict, f)
        with open(args.target_label_path, 'w') as f:
            json.dump(target_label_dict, f)
        
        return examples, source_label_dict, target_label_dict
    else:
        return examples


def generate_source_and_target_data(examples, source_label_dict, target_label_dict):
    source_examples, target_examples = list(), list()
    for example in examples:
        event_type_set = set([event["event_type"] for event in example.event_list])
        if set(target_label_dict).intersection(event_type_set) and set(target_label_dict).intersection(event_type_set) != {'None'}:
            for event in example.event_list:
                if event['event_type'] not in target_label_dict:
                    event['event_type'] = "None"
                    event['label_id'] = 0
            target_examples.append(example)
        elif event_type_set == {'None'}:
            source_examples.append(example)
            target_examples.append(example)
        else:
            source_examples.append(example)

    for target_example in target_examples:
        for event in target_example.event_list:
            event["label_id"] = target_label_dict[event["event_type"]]
    for source_example in source_examples:
        for event in source_example.event_list:
            event["label_id"] = source_label_dict[event["event_type"]]

    return source_examples, target_examples


def generate_few_shot_target_data(args, target_examples, target_event_type_list, sample_th):
    filtered_examples = defaultdict(list)
    event_type_counter = defaultdict(int)

    shuffle(target_examples)
    filtered_example_idx_list = list()
    for event_type in tqdm(target_event_type_list):
        examples_this_turn = [(idx, example) for idx, example in enumerate(target_examples) if idx not in filtered_example_idx_list]
        for idx, example in examples_this_turn:
            if event_type_counter[event_type] >= sample_th:
                break
            event_counter_this_sent = get_counter_this_sent(example)
            if event_type in event_counter_this_sent:
                filtered_example_idx_list.append(idx)
                filtered_examples[event_type].append(example)
                for event_type_this_sent, cnt in event_counter_this_sent.items():
                    event_type_counter[event_type_this_sent] += cnt

    post_process_filtered_examples = post_process(filtered_examples, event_type_counter, sample_th)
    return post_process_filtered_examples
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="ACE", type=str, choices=['ACE', 'MAVEN', 'ERE'])
    parser.add_argument("--train_K", default=5, type=int)
    parser.add_argument("--dev_K", default=2, type=int)
    parser.add_argument("--only_few_shot", action='store_true', default=False)
    config = parser.parse_args()

    if config.dataset_type=='ACE':
        file_path = "../ACE05_processed/train.json"
        file_test_path = "../ACE05_processed/test.json"
        target_event_type_num = 23
        args = edict(
            label_dict_path="../k_shot/label_dict_ACE.json", 
            dataset_type="ACE", 
            source_label_path = "./label_dict_ACE-source.json",
            target_label_path = "./label_dict_ACE-target.json",
        )
    elif config.dataset_type=='MAVEN':
        file_path = "../MAVEN/train.jsonl"
        file_test_path = "../MAVEN/valid.jsonl"
        target_event_type_num = 48
        args = edict(
            label_dict_path="../k_shot/label_dict_MAVEN.json", 
            dataset_type="MAVEN",
            source_label_path = "./label_dict_MAVEN-source.json",
            target_label_path = "./label_dict_MAVEN-target.json",
        )
    elif config.dataset_type=='ERE':
        file_path = "../ERE/train.jsonl"
        file_test_path = "../ERE/test.jsonl"
        target_event_type_num = 28
        args = edict(
            label_dict_path="../k_shot/label_dict_ERE.json", 
            dataset_type="ERE", 
            source_label_path = "./label_dict_ERE-source.json",
            target_label_path = "./label_dict_ERE-target.json",
        )       
    else:
        raise AssertionError()
    
    # Training dataset
    examples, source_label_dict, target_label_dict = generate_data(args, file_path, target_event_type_num=target_event_type_num)
    source_examples, target_examples = generate_source_and_target_data(examples, source_label_dict, target_label_dict)

    if not config.only_few_shot:
        source_examples_dict_train, source_examples_dict_dev = defaultdict(list), defaultdict(list)
        for example in source_examples:
            event_type = example.event_list[0]["event_type"]
            if uniform(0, 1) < 0.8:
                source_examples_dict_train[event_type].append(example)
            else:
                source_examples_dict_dev[event_type].append(example)

        output_source_train_path = "./fewshot_set/{}_source_train".format(config.dataset_type)
        if not os.path.exists(output_source_train_path):
            os.makedirs(output_source_train_path, exist_ok=True)
        data_split_and_save(source_examples_dict_train, output_source_train_path, set_type="train")
        output_source_dev_path = "./fewshot_set/{}_source_dev".format(config.dataset_type)
        if not os.path.exists(output_source_dev_path):
            os.makedirs(output_source_dev_path, exist_ok=True)
        data_split_and_save(source_examples_dict_dev, output_source_dev_path, set_type="dev")

        output_target_path = "./fewshot_set/{}_target_train.pkl".format(config.dataset_type)
        pickle.dump(target_examples, open(output_target_path, "wb"))
    
    seed_list = [13, 21, 42, 88, 100, 18, 26, 47, 93, 105]
    for i, seed in enumerate(seed_list):
        set_seed(seed)
        output_target_path = "./fewshot_set/K{}_{}_t{}".format(config.train_K, config.dataset_type, i)
        if not os.path.exists(output_target_path):
            os.makedirs(output_target_path, exist_ok=True)
        # target-train
        filtered_target_examples = generate_few_shot_target_data(args, target_examples, target_label_dict, sample_th=config.train_K)
        data_split_and_save(filtered_target_examples, output_target_path, set_type="train")
        # target-dev
        filtered_target_examples = generate_few_shot_target_data(args, target_examples, target_label_dict, sample_th=config.dev_K)
        data_split_and_save(filtered_target_examples, output_target_path, set_type="dev")

    if not config.only_few_shot:
        # Test dataset
        examples = generate_data(args, file_test_path)
        source_examples, target_examples = generate_source_and_target_data(examples, source_label_dict, target_label_dict)

        source_examples_dict = defaultdict(list)
        for example in source_examples:
            event_type = example.event_list[0]["event_type"]
            source_examples_dict[event_type].append(example)
        output_source_path = "./fewshot_set/{}_source_test".format(config.dataset_type)
        data_split_and_save(source_examples_dict, output_source_path, set_type="test")

        target_examples_dict = defaultdict(list)
        for example in target_examples:
            event_type = example.event_list[0]["event_type"]
            target_examples_dict[event_type].append(example)
        output_target_path = "./fewshot_set/{}_target_test".format(config.dataset_type)
        data_split_and_save(target_examples_dict, output_target_path, set_type="test")