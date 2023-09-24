import json
import jsonlines
import logging
import torch
import random
import numpy as np

from torch.utils.data import DataLoader, SequentialSampler
from processor.sample import SentDataset

logger = logging.getLogger(__name__)


def set_seed(args):
    if isinstance(args, int):
        random.seed(args)
        np.random.seed(args)
        torch.manual_seed(args)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
        
def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def read_jsonlines(file_path):
    lines = list()
    with jsonlines.open(file_path) as reader:
        for line in reader:
            lines.append(line)
    return lines


def construct_prompt(label_dict, tokenizer, verbose=False):
    input_ids_list, attention_mask_list = list(), list()
    start_list, end_list = list(), list()

    for event_type, event_id in sorted(label_dict.items(), key=lambda x:x[1]):
        start, end = None, None
        if event_id==0:
            prompt = "It is an unknown event or non-event."
        else:
            event_type = event_type.replace("_", " ")
            event_type = event_type.replace("-", " ")
            prompt_start, prompt_end = event_type.split()[0], event_type.split()[-1]
            prompt = "It is an event about {}".format(event_type)
    
        text_tokens = [tokenizer.cls_token]              
        for word in prompt.split():
            if event_id!=0 and word==prompt_start:
                start = len(text_tokens)
            token = tokenizer.tokenize(" "+word)
            for sub_token in token:
                text_tokens.append(sub_token)
            if event_id!=0 and word==prompt_end:
                end = len(text_tokens)       
        text_tokens.append(tokenizer.sep_token)
        if event_id==0:
            start, end = 0, len(text_tokens)-1
        assert(start is not None and end is not None and start<end)
        start_list.append(start)
        end_list.append(end)

        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        input_ids_list.append(input_ids)
    
    max_input_length = max([len(input_ids) for input_ids in input_ids_list])
    for input_ids in input_ids_list:
        attention_mask_list.append(
            [1]*len(input_ids)+[0]*(max_input_length-len(input_ids))
        )
        input_ids.extend([tokenizer.pad_token_id]*(max_input_length-len(input_ids)))

    if verbose:
        for input_ids, start, end in zip(input_ids_list, start_list, end_list):
            print("--------------------------")
            print("Prompt: {}\tEvent: {}".format(
                tokenizer.decode(input_ids), tokenizer.decode(input_ids[start:end])
            ))

    return torch.tensor(input_ids_list, dtype=torch.long), \
            torch.tensor(attention_mask_list, dtype=torch.long), \
            start_list, end_list
            
            
def compute_embeddings(model, sent_features, processor, args, verbose=True):
    dataset = processor.convert_features_to_dataset_sent(sent_features)
    sent_dataloader = DataLoader(
        dataset, 
        sampler=SequentialSampler(dataset),
        batch_size=args.eval_batch_size,
        collate_fn=SentDataset.collate_fn,
    )

    label_list = []
    if verbose:
        logger.info("Start computing features")
    tok_embed_list = list()

    model.eval()
    with torch.no_grad():
        for batch in sent_dataloader:
            inputs = {
                    'input_ids':       batch[0].to(args.device),
                    'attention_mask':  batch[1].to(args.device), 
                    'start_list_list': [start_list.to(args.device) for start_list in batch[3]],
                    'end_list_list':   [end_list.to(args.device) for end_list in batch[4]],
                    'compute_features_only': True,
                }
            tok_embeds = model(**inputs)
            tok_embed_list.append(tok_embeds.detach().cpu().numpy())
            label_list.extend(torch.cat(batch[5]).tolist())

    tok_embeddings = np.concatenate(tok_embed_list, axis=0)
    if verbose:
        logger.info("End computing features")

    sent_features = [sent_feature for sent_feature in sent_features if sent_feature is not None]
    event_feature_num = (sent_features[-1].feature_idx_list[-1]+1)
    assert(tok_embeddings.shape[0]==event_feature_num)
    return tok_embeddings


def cut_pred_res(pred_labels, gt_res_list):
    gt_len_list = [len(gt_res) for gt_res in gt_res_list]
    assert(sum(gt_len_list)==len(pred_labels))

    pred_res_list = list()
    curr = 0
    for gt_len in gt_len_list:
        pred_res = pred_labels[curr:(curr+gt_len)]
        pred_res_list.append(pred_res)
        curr += gt_len
    return pred_res_list


def get_interval_res(label_idx_list, start_list=None):
    if isinstance(label_idx_list, torch.Tensor):
        label_idx_list = label_idx_list.tolist()
    res_list = []
    last_label_idx = 0
    start, end = None, None

    if start_list is None:
        word_idx_list = range(len(label_idx_list))
    else:
        word_idx_list = list()
        curr_idx, last_v = -1, -1
        for v in start_list:
            if v!=last_v:
                curr_idx += 1
            word_idx_list.append(curr_idx)
            last_v = v

    for idx, label_idx in zip(word_idx_list, label_idx_list):
        if last_label_idx==0 and label_idx!=0:
            start = idx
        elif last_label_idx!=0 and label_idx==0:
            end = idx
            res_list.append((start, end, last_label_idx))
            start, end = None, None
        elif last_label_idx!=0 and label_idx!=0 and label_idx!=last_label_idx:
            end = idx if idx!=start else (start+1)
            res_list.append((start, end, last_label_idx))
            start, end = None, None
            start = idx
        last_label_idx = label_idx

    if start is not None and last_label_idx:
        end = idx+1
        res_list.append((start, end, last_label_idx))
    
    return res_list


def eval_score_seq_label(gt_list, pred_list):
    gt_num, pred_num, correct_num = 0, 0, 0
    for (gt, pred) in zip(gt_list, pred_list):
        gt = set(gt); pred = set(pred)
        gt_num += len(gt)
        pred_num += len(pred)
        correct_num += len(gt.intersection(pred))
    recall = correct_num/gt_num if gt_num!=0 else 0
    precision = correct_num/pred_num if pred_num!=0 else 0
    f1 = 2*recall*precision/(recall + precision) if correct_num!=0 else 0
    return recall, precision, f1, gt_num, pred_num, correct_num  


def show_results(features, gt_list, pred_list, processor, output_path):
    labels_dict = {v:k for k,v in processor.label_dict.items()}
    features = [feature for feature in features if feature is not None]
    assert(len(features)==len(gt_list))
    assert(len(features)==len(pred_list))

    with open(output_path, 'w') as f:
        for feature, gt, pred in zip(features, gt_list, pred_list):
            sent = feature.sent
            f.write("--------------------------------------------------------------------\n")
            f.write("{}\n".format(" ".join(sent)))

            all_res = dict()
            for start, end, label in gt:
                if (start, end) not in all_res:
                    all_res[(start, end)] = {"gt":0, "pred":0}
                all_res[(start, end)]["gt"] = int(label)
            for start, end, label in pred:
                if (start, end) not in all_res:
                    all_res[(start, end)] = {"gt":0, "pred":0}
                all_res[(start, end)]["pred"] = int(label)
            
            for (start, end), res in all_res.items():
                if res["gt"]==0 and res["pred"]==0:
                    continue
                elif res["gt"]==res["pred"]:
                    f.write("Trigger {} Matched. Gt type: {}. Pred type: {}.\n".format(
                        " ".join(sent[start:end]), labels_dict[res["gt"]], labels_dict[res["pred"]])
                    )
                else:
                    f.write("Trigger {} Dismatched. Gt type: {}. Pred type: {}.\n".format(
                        " ".join(sent[start:end]), labels_dict[res["gt"]], labels_dict[res["pred"]])
                    )


def get_label_mask(query_label_ids, key_label_ids):
    """
    It must be ensured that the samples in last K columns in queue is exactly the query.
    """
    L_q, L_k = query_label_ids.size(0), key_label_ids.size(0)

    numerator_label_mask = (query_label_ids[:, None]==key_label_ids[None, :]).float()
    denominator_label_mask = torch.ones_like(numerator_label_mask)

    numerator_label_mask[torch.arange(L_q), torch.arange(L_k-L_q, L_k)] = 0.0
    denominator_label_mask[torch.arange(L_q), torch.arange(L_k-L_q, L_k)] = 0.0

    return numerator_label_mask, denominator_label_mask


def _loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    n = mu_i.shape[0]
    m = mu_j.shape[0]

    mu_i = mu_i.unsqueeze(1).expand(n,m, -1)
    sigma_i = sigma_i.unsqueeze(1).expand(n,m,-1)
    mu_j = mu_j.unsqueeze(0).expand(n,m,-1)
    sigma_j = sigma_j.unsqueeze(0).expand(n,m,-1)
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 2)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=2)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=2)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 2)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=2)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=2)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d