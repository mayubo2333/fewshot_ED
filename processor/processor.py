import sys
sys.path.append("/mnt/lustre/ybma/")
import logging

import pickle
from tqdm import tqdm
from random import shuffle, sample
from collections import defaultdict, Counter
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from fewshot_ED.utils import read_json, read_jsonlines
from fewshot_ED.processor.sample import Event, Sent, SentFeatures, EventFeatures, SentDataset

logger = logging.getLogger(__name__)                         


class EventDetectionProcessor:
    def __init__(self, args, tokenizer):
        self.args = args
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.label_dict = read_json(self.args.label_dict_path)
        self.train_labeled_sent = list()      


    def create_examples_maven(self, lines, set_type):
        sent_dict = {}
        examples = []
        for line in tqdm(lines):
            text_list = [content["tokens"] for content in line["content"]]
            for event in line["events"]:
                event_type = event["type"]
                trigger = {
                    "text": event["mention"][0]["trigger_word"],
                    "start": event["mention"][0]["offset"][0],
                    "end": event["mention"][0]["offset"][1],
                    "label_id": self.label_dict[event_type],
                }
                sent = text_list[event["mention"][0]["sent_id"]]
                if " ".join(sent) not in sent_dict:
                    sent_dict[" ".join(sent)] = len(sent_dict)
                    if set_type=="train" and " ".join(sent) not in self.train_labeled_sent:
                        self.train_labeled_sent.append(" ".join(sent))

                examples.append(
                    Event(sent_dict[" ".join(sent)], sent, event_type, trigger)
                )

            for negative in line["negative_triggers"]:
                event_type = "None"
                trigger = {
                    "text": negative["trigger_word"],
                    "start": negative["offset"][0],
                    "end": negative["offset"][1],
                    "label_id": self.label_dict[event_type],
                }
                sent = text_list[negative["sent_id"]]
                if " ".join(sent) not in sent_dict:
                    sent_dict[" ".join(sent)] = len(sent_dict)
                    if set_type=="train" and " ".join(sent) not in self.train_labeled_sent:
                        self.train_labeled_sent.append(" ".join(sent))
                examples.append(
                    Event(sent_dict[" ".join(sent)], sent, event_type, trigger)
                )
        return examples


    def create_examples_ere(self, lines, set_type):
        sent_dict = {}
        positive_idx_dict = defaultdict(list)
        examples = []
        for line in tqdm(lines):
            sent = line["word_list"]
            if " ".join(sent) not in sent_dict:
                sent_dict[" ".join(sent)] = len(sent_dict)
                if set_type=="train" and " ".join(sent) not in self.train_labeled_sent:
                    self.train_labeled_sent.append(" ".join(sent))
                    
            event_type = line["event_type"]
            start, end = line["span"]
            for i in range(start, end):
                trigger = {
                    "text": sent[i],
                    "start": i,
                    "end": i+1,
                    "label_id": self.label_dict[event_type],
                }
                positive_idx_dict[" ".join(sent)].append(i)
                examples.append(
                    Event(sent_dict[" ".join(sent)], sent, event_type, trigger)
                )

        for text in sent_dict:
            sent_idx = sent_dict[text]
            positive_idx_list = positive_idx_dict[text]
            sent = text.split()
            for i in range(len(sent)):
                if i not in positive_idx_list:
                    event_type = "None"
                    event_trigger = {
                        "text": sent[i],
                        "start": i,
                        "end": i+1,
                        "label_id": self.label_dict[event_type]
                    }
                    examples.append(Event(sent_idx, sent, event_type, event_trigger))
        return examples


    def create_examples_ace(self, lines, set_type):
        sent_dict = {}
        positive_idx_dict = defaultdict(list)
        examples = []
        for line in tqdm(lines):  
            sent = [word.replace(" ", "") for word in line['tokens']]
            if " ".join(sent) not in sent_dict:
                sent_dict[" ".join(sent)] = len(sent_dict)
                if set_type=="train" and " ".join(sent) not in self.train_labeled_sent:
                    self.train_labeled_sent.append(" ".join(sent))
            event_type = line['event_type']
            if event_type != "None":
                start, end = line["trigger_start"], line["trigger_end"]+1
                for i in range(start, end):
                    event_trigger = {
                        "text": sent[i],
                        "start": i,
                        "end": i+1,
                        "label_id": self.label_dict[event_type]
                    }
                    positive_idx_dict[" ".join(sent)].append(i)
                    examples.append(Event(sent_dict[" ".join(sent)], sent, event_type, event_trigger))
        
        for text in sent_dict:
            sent_idx = sent_dict[text]
            positive_idx_list = positive_idx_dict[text]
            sent = text.split()
            for i in range(len(sent)):
                if i not in positive_idx_list:
                    event_type = "None"
                    event_trigger = {
                        "text": sent[i],
                        "start": i,
                        "end": i+1,
                        "label_id": self.label_dict[event_type]
                    }
                    examples.append(Event(sent_idx, sent, event_type, event_trigger))

        return examples


    def create_examples(self, lines, set_type=None):
        if 'MAVEN' in self.args.dataset_type:
            event_examples = self.create_examples_maven(lines, set_type) 
        elif self.args.dataset_type=='ERE':
            event_examples = self.create_examples_ere(lines, set_type)
        else:
            event_examples = self.create_examples_ace(lines, set_type)

        event_list_dict = defaultdict(list)
        for event in event_examples:
            sent_idx = event.sent_idx
            event_list_dict[sent_idx].append(event)
        sent_examples = list()
        for i in range(len(event_list_dict)):
            event_list = event_list_dict[i]
            sent_example = Sent(event_list)
            sent_examples.append(sent_example)

        return event_examples, sent_examples
    

    def convert_examples_to_features_sent(self, examples, set_type):
        sent_features_dict = dict()
        feature_counter = 0
        for example in tqdm(examples):
            sent_idx, sent = example.sent_idx, example.sent
            event_list = example.event_list
            try:
                sent_feature = SentFeatures(sent_idx, sent, event_list, self.tokenizer, self.args.max_seq_length, feature_counter)
                feature_counter += len(event_list)
            except AssertionError:
                if set_type != "test":
                    sent_feature = None
                else:
                    print("Sentence in test set exceeding the max length constraint")
                    # raise AssertionError("Sentence in test set exceeding the max length constraint")
            sent_features_dict[sent_idx] = sent_feature
        
        sent_features_list = [sent_feature for _, sent_feature in sorted(sent_features_dict.items(), key=lambda x:x[0])]
        return sent_features_list


    def convert_features_to_dataset_sent(self, sent_features):
        return SentDataset(sent_features)


    def convert_examples_to_features_event(self, sent_feature_list):
        features = list()
        for sent_feature in sent_feature_list:
            if sent_feature is None:
                continue
            else: 
                sent = sent_feature.sent
                enc_input_ids, enc_mask_ids = sent_feature.enc_input_ids, sent_feature.enc_mask_ids
                for event_type, trigger, start, end, label, feature_idx in sent_feature.wrapped_events: 
                    feature = EventFeatures(
                        feature_idx, event_type, trigger, sent, 
                        enc_input_ids, enc_mask_ids, 
                        start, end, label
                    )
                    features.append(feature)
        return features


    def convert_features_to_dataset_event(self, features):
        all_input_ids = torch.tensor([f.enc_input_ids for f in features], \
            dtype=torch.long)
        all_mask_ids = torch.tensor([f.enc_mask_ids for f in features], \
            dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], \
            dtype=torch.long)
        all_feature_idx = torch.tensor([f.feature_idx for f in features], \
            dtype=torch.long)
        all_tok_start = torch.tensor([f.tok_start for f in features], \
            dtype=torch.long)
        all_tok_end = torch.tensor([f.tok_end for f in features], \
            dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_mask_ids, all_label_ids, all_feature_idx, all_tok_start, all_tok_end)
        return dataset


    def generate_features(self, set_type):
        assert (set_type in ['train', 'dev', 'test'])
        if set_type=='train':
            file_path = self.args.train_file
        elif set_type=='dev':
            file_path = self.args.dev_file
        else:
            file_path = self.args.test_file
        
        if file_path.endswith("pkl"):
            event_examples = None
            sent_examples = list()
            sent_examples_dict = pickle.load(open(file_path, "rb"))
            for event_type, examples_this_type in sent_examples_dict.items():
                sent_examples.extend(examples_this_type)
        else:
            lines = read_jsonlines(file_path) if file_path.endswith("jsonl") else read_json(file_path)
            event_examples, sent_examples = self.create_examples(lines, set_type)
        #Re-index
        for idx, sent_example in enumerate(sent_examples):
            sent_example.sent_idx = idx

        sent_features = self.convert_examples_to_features_sent(sent_examples, set_type=set_type)
        event_features = self.convert_examples_to_features_event(sent_features)

        logger.info("{} events in {} sentences are collected.".format(len(event_features), len(sent_features)))
        return event_examples, sent_examples, event_features, sent_features


    def generate_sent_dataloader(self, set_type):
        assert (set_type in ['train', 'dev', 'test'])
        _, sent_examples, _, sent_features = self.generate_features(set_type)
        
        batch_size = self.args.batch_size if set_type=="train" else self.args.eval_batch_size
        dataset = self.convert_features_to_dataset_sent(sent_features)
        sampler=SequentialSampler(dataset) if 'train' not in set_type else RandomSampler(dataset)
        sent_dataloader = DataLoader(
            dataset, 
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=SentDataset.collate_fn,
            drop_last=(self.args.drop_last and set_type=="train"),
        )
        return sent_examples, sent_features, sent_dataloader

    
    def generate_event_dataloader(self, set_type, sent_features=None):
        event_examples = None
        if sent_features is not None:
            assert(isinstance(sent_features[0], SentFeatures))
            event_features = self.convert_examples_to_features_event(sent_features)
        else:
            event_examples, _, event_features, _ = self.generate_features(set_type)
        dataset = self.convert_features_to_dataset_event(event_features)

        batch_size = self.args.batch_size if set_type=="train" else self.args.eval_batch_size
        event_dataloader = DataLoader(
            dataset, 
            sampler=SequentialSampler(dataset) if 'train' not in set_type else RandomSampler(dataset),
            batch_size=batch_size,
        )
        return event_examples, event_features, event_dataloader


    def generate_tenure_features(self, sent_features, min_sample_per_class=4):
        event_features = self.convert_examples_to_features_event(sent_features)
        shuffle(event_features)
        sampled_event_features, eventtype_featureid_dict = list(), defaultdict(list)

        for feature in event_features:
            event_type = feature.event_type
            if event_type == "None":
                continue
            if len(eventtype_featureid_dict[event_type])<min_sample_per_class:
                eventtype_featureid_dict[event_type].append(len(sampled_event_features))
                sampled_event_features.append(feature)

        event_dataset = self.convert_features_to_dataset_event(sampled_event_features)
        event_dataloader = DataLoader(event_dataset, shuffle=False, batch_size=self.args.eval_batch_size)
        return sampled_event_features, event_dataloader

    
    def generate_trigger_dict(self, sent_features, set_type, count=False):
        trigger_dict = defaultdict(list)
        for sent_feature in sent_features:
            if sent_feature is None:
                continue
            trigger_list = [event["text"] for event in sent_feature.event_list]    
            for (trigger, label) in zip(trigger_list, sent_feature.label_list  ):
                if label==0:
                    continue
                trigger_dict[label].append(trigger)
        if count:
            trigger_dict = {label:tuple(Counter(trigger_list).most_common()) for label, trigger_list in trigger_dict.items()}
        else:
            trigger_dict = {label:tuple(list(set(trigger_list))) for label, trigger_list in trigger_dict.items()}
        return trigger_dict