import jsonlines
import ipdb
from tqdm import tqdm
from collections import defaultdict


def convert_char_id_to_word_id(text):
    word_list = text.split()
    start_pos_dict, end_pos_dict = dict(), dict()
    curr_pos = 0

    for idx, word in enumerate(word_list):
        start_pos_dict[curr_pos] = idx
        curr_pos += len(word)
        end_pos_dict[curr_pos] = idx
        curr_pos += 1
    return start_pos_dict, end_pos_dict

set_type_list = ["train", "dev", "test"]
split_dict = dict()
for set_type in set_type_list:
    with open("./splits/{}.doc.txt".format(set_type)) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            split_dict[line] = set_type

train_data, dev_data, test_data = list(), list(), list()
data_class = {
    "train": train_data,
    "dev": dev_data,
    "test": test_data,
}


data_file_list = [
    "./processed/LDC2015E29.unified.jsonl",
    "./processed/LDC2015E68.unified.jsonl",
    "./processed/LDC2015E78.unified.jsonl"
]
lines = list()
for data_file in data_file_list:
    with jsonlines.open(data_file) as reader:
        for line in reader:
            lines.append(line)

dropped_sent, dropped_event = 0, 0
for line in tqdm(lines):
    text = line["text"]
    doc_id = line["id"].split("-")[0]
    try:
        set_type = split_dict[doc_id]
    except:
        dropped_sent += 1
        continue

    start_pos_dict, end_pos_dict = convert_char_id_to_word_id(text)
    trigger_loc_dict = defaultdict(list)
    candidate_event_dict = dict()
    for event in line["events"]:
        for mention in event["triggers"]:
            start, end = mention["position"]
            try:
                start_word, end_word = start_pos_dict[start], end_pos_dict[end]+1
            except:
                dropped_event += 1
                continue
                   
            new_line = {
                "doc_id": line["id"],
                "word_list": text.split(),
                "trigger_text": mention["trigger_word"],
                "span": (start_word, end_word),
            }
            trigger_loc_dict[(start_word, end_word)].append(event["type"])
            candidate_event_dict[(start_word, end_word)] = new_line
    
    for (start_word, end_word), event_type_list in trigger_loc_dict.items():
        event_type_list = sorted(list(set(event_type_list)))
        if 'transferownership' in event_type_list and 'transfermoney' in event_type_list:
            event_type_list = ['transferownership']
        if 'die' in event_type_list and 'execute' in event_type_list:
            event_type_list = ['execute']

        new_line = candidate_event_dict[(start_word, end_word)]
        new_line["event_type"] = event_type_list[0]
        data_class[set_type].append(new_line)
 
    
    for negative_event in line["negative_triggers"]:
        start, end = negative_event["position"]
        try:
            start_word, end_word = start_pos_dict[start], end_pos_dict[end]+1
        except:
            dropped_event += 1
            continue
        new_line = {
            "doc_id": line["id"],
            "word_list": text.split(),
            "event_type": "None",
            "trigger_text": negative_event["trigger_word"],
            "span": (start_word, end_word),
        }
        data_class[set_type].append(new_line)
    

for set_type in set_type_list:
    with jsonlines.open("{}.jsonl".format(set_type), 'w') as writer:
        for line in data_class[set_type]:
            writer.write(line)