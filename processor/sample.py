import logging
from copy import deepcopy

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