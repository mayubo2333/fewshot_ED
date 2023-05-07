import sys
sys.path.append("../")
import logging

from tqdm import tqdm
from collections import defaultdict

from utils import read_json
from processor.sample import Event, Sent

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
                if set_type=="train_unlabeled" and " ".join(sent) in self.train_labeled_sent:
                    continue
                if " ".join(sent) not in sent_dict:
                    sent_dict[" ".join(sent)] = len(sent_dict)
                    if set_type=="train_labeled" and " ".join(sent) not in self.train_labeled_sent:
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
                if set_type=="train_unlabeled" and " ".join(sent) in self.train_labeled_sent:
                    continue
                if " ".join(sent) not in sent_dict:
                    sent_dict[" ".join(sent)] = len(sent_dict)
                    if set_type=="train_labeled" and " ".join(sent) not in self.train_labeled_sent:
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
            if set_type=="train_unlabeled" and " ".join(sent) in self.train_labeled_sent:
                continue
            if " ".join(sent) not in sent_dict:
                sent_dict[" ".join(sent)] = len(sent_dict)
                if set_type=="train_labeled" and " ".join(sent) not in self.train_labeled_sent:
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
            if set_type=="train_unlabeled" and " ".join(sent) in self.train_labeled_sent:
                continue
            if " ".join(sent) not in sent_dict:
                sent_dict[" ".join(sent)] = len(sent_dict)
                if set_type=="train_labeled" and " ".join(sent) not in self.train_labeled_sent:
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