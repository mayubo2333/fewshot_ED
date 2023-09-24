import logging
from copy import deepcopy

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Event:
    def __init__(self, sent_idx, sent, event_type, trigger):
        self.sent_idx = sent_idx
        self.sent = sent
        self.event_type = event_type
        self.trigger = trigger
    
    def __repr__(self):
        s = ""
        s += "sent {}: {}\n".format(self.sent_idx, self.sent)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger: {}\n".format(self.trigger)
        return s


class Sent:
    def __init__(self, event_list):
        self.sent_idx = event_list[0].sent_idx
        self.sent = event_list[0].sent
        self.event_list = []
        for event in event_list:
            event_ = deepcopy(event.trigger)
            event_.update({"event_type": event.event_type})
            self.event_list.append(event_)
        self.filter_repeat()
        self.sort_event_list()
    
    def sort_event_list(self):
        self.event_list = sorted(self.event_list, key=lambda x:(x["start"], x["end"]))
    
    def filter_repeat(self):
        filtered_event_list = list()
        key_list = list()
        for event in self.event_list:
            key = "_".join([event['text'], str(event['start']), str(event['end']), str(event['label_id'])])
            if key not in key_list:
                key_list.append(key)
                filtered_event_list.append(event)
        self.event_list = filtered_event_list

    def __repr__(self):
        s = ""
        s += "sent {}: {}\n".format(self.sent_idx, self.sent)
        s += "event_list: {}\n".format(self.event_list)
        return s
    

class EventFeatures:
    """A single set of features of data."""

    def __init__(self, feature_idx, event_type, event_trigger, sent, 
        enc_input_ids, enc_mask_ids, 
        tok_start, tok_end, label_id,
    ):
        self.feature_idx = feature_idx
        self.event_type = event_type
        self.event_trigger = event_trigger
        self.sent = sent
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.tok_start = tok_start
        self.tok_end = tok_end
        self.label_id = label_id


    def __repr__(self):
        s = "" 
        s += "feature_idx: {}\n".format(self.feature_idx)
        s += "event_type: {}\n".format(self.event_type)
        s += "event_trigger: {}\n".format(self.event_trigger)
        s += "sent: {}\n".format(self.sent)

        s += "input_ids: {}\n".format(self.enc_input_ids)
        s += "mask_ids: {}\n".format(self.enc_mask_ids)
        s += "tok_start: {}\tend:{}\n".format(self.tok_start, self.tok_end)
        s += "label_id: {}\n".format(self.label_id)
        return s


class SentFeatures:
    def __init__(self, sent_idx, sent, event_list, tokenizer, max_seq_length, feature_counter):
        self.sent_idx = sent_idx
        self.sent = sent if isinstance(sent, list) else sent.split()
        self.event_list = event_list
        self.tokenizer = tokenizer

        enc_input_ids, enc_mask_ids, word_to_token_index = self.build_index()
        self.padding_sent(enc_input_ids, enc_mask_ids, max_seq_length)

        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.word_to_token_index = word_to_token_index
        self.start_list = [word_to_token_index[event["start"]] for event in event_list]
        self.end_list = [word_to_token_index[event["end"]] for event in event_list]
        self.label_list = [event["label_id"] for event in event_list]
        self.event_type_list = [event["event_type"] for event in event_list]
        self.feature_idx_list = list(range(feature_counter, feature_counter+len(event_list)))
        
        
    @property
    def wrapped_events(self):
        trigger_list = [self.tokenizer.decode(self.enc_input_ids[start:end]) for (start, end) in zip(self.start_list, self.end_list)]
        return zip(self.event_type_list, trigger_list, self.start_list, self.end_list, self.label_list, self.feature_idx_list)
    

    def __repr__(self):
        s = ""
        s += "sent {}: {}\n".format(self.sent_idx, self.sent)
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        for start, end, event_type in zip(self.start_list, self.end_list, self.event_type_list):
            s += "Event type: {}, Trigger: {}\n".format(
                event_type, self.tokenizer.decode(self.enc_input_ids[start:end])
            )
        return s


    def padding_sent(self, enc_input_ids, enc_mask_ids, max_seq_length):
        if len(enc_input_ids) > max_seq_length:
            logger.info("Sentence {} with length {} dropped due to exceeding {} constraint length: {}".format(
            # print("Sentence {} with length {} dropped due to exceeding {} constraint length: {}".format(
                        self.sent_idx, len(enc_input_ids), " ".join(self.sent), max_seq_length)
            )
            raise AssertionError()
        while len(enc_input_ids) < max_seq_length:
            enc_input_ids.append(self.tokenizer.pad_token_id)
            enc_mask_ids.append(0)


    def build_index(self):
        word_to_char_index, curr = list(), 0
        for word in self.sent:
            word_to_char_index.append(curr)
            curr += len(word)+1
        word_to_char_index.append(curr)

        word_to_token_index = list()
        enc_text = " ".join(self.sent)
        enc = self.tokenizer(enc_text)
        enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]

        for i in range(len(word_to_char_index)-1):
            char_idx = word_to_char_index[i]
            token_idx = enc.char_to_token(char_idx)
            word_to_token_index.append(token_idx)
        word_to_token_index.append(sum(enc_mask_ids)-1)

        return enc_input_ids, enc_mask_ids, word_to_token_index


class SentDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
 
    @staticmethod
    def collate_fn(batch):
        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch if f is not None], dtype=torch.long)
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch if f is not None], dtype=torch.long)
        sent_idx = torch.tensor([f.sent_idx for f in batch if f is not None], dtype=torch.long)

        tok_start_list = [torch.tensor(f.start_list, dtype=torch.long) for f in batch if f is not None]
        tok_end_list = [torch.tensor(f.end_list, dtype=torch.long) for f in batch if f is not None]
        label_list = [torch.tensor(f.label_list, dtype=torch.long) for f in batch if f is not None] 
        feature_idx_list = [torch.tensor(f.feature_idx_list, dtype=torch.long) for f in batch if f is not None] 

        return enc_input_ids, enc_mask_ids, sent_idx, tok_start_list, tok_end_list, label_list, feature_idx_list