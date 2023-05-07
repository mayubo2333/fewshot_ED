import json
import jsonlines
import argparse


def get_maven_label_dict():
    file_input_path, file_output_path = "../MAVEN/train.jsonl", "./label_dict_MAVEN.json"
    label_dict = {"None": 0}

    with jsonlines.open(file_input_path) as reader:
        for line in reader:
            for event in line["events"]:
                event_type, type_id = event["type"], event["type_id"]
                label_dict[event_type] = type_id

    with open(file_output_path, 'w') as f:
        json.dump(label_dict, f)


def get_ace_label_dict():
    file_input_path, file_output_path = "../ACE05_processed/train.json", "./label_dict_ACE.json"
    label_dict = {"None": 0}

    with open(file_input_path) as f:
        lines = json.load(f)
    for line in lines:  
        event_type = line['event_type']
        if event_type not in label_dict:
            label_dict[event_type] = len(label_dict)

    with open(file_output_path, 'w') as f:
        json.dump(label_dict, f)


def get_ere_label_dict():
    file_input_path, file_output_path = "../ERE/train.jsonl", "./label_dict_ERE.json"
    label_dict = {"None": 0}
    with jsonlines.open(file_input_path) as reader:
        for line in reader:  
            event_type = line["event_type"]
            if event_type not in label_dict:
                label_dict[event_type] = len(label_dict)

    with open(file_output_path, 'w') as f:
        json.dump(label_dict, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="ACE", type=str, choices=[
        'ACE', 'MAVEN', 'ERE'
    ])
    args = parser.parse_args()
    
    if args.dataset_type=="ACE":
        get_ace_label_dict()
    elif args.dataset_type=="MAVEN":
        get_maven_label_dict()
    elif args.dataset_type=="ERE":
        get_ere_label_dict()
    else:
        raise AssertionError()