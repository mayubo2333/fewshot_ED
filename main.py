import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
import ipdb
import json
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict
from transformers import RobertaConfig, RobertaTokenizerFast
from torch.cuda.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from processor.processor import EventDetectionProcessor
from models.model import EventDetectionModel
from models.momentum_cl import Momentum_CL
from models.batch_cl import Batch_CL
from utils import read_json, set_seed, show_results, construct_prompt, compute_embeddings, cut_pred_res, get_interval_res, eval_score_seq_label
from crf.crf_inference import get_abstract_transitions, ViterbiDecoder

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, EventDetectionModel, RobertaTokenizerFast),
}

def train(args, model, processor):
    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader = processor.generate_sent_dataloader('train')
    train_iter = iter(train_dataloader)
    logger.info("dev dataloader generation")
    _, dev_features, dev_dataloader = processor.generate_sent_dataloader('dev')
    logger.info("test dataloader generation")
    _, test_features, test_dataloader = processor.generate_sent_dataloader('test') 

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.max_steps*args.warmup_steps), num_training_steps=args.max_steps)
    if args.fp_16:
        scaler = GradScaler()

    # Train!
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'event'))
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  batch size = %d", train_dataloader.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step, step = -1, -1
    smooth_loss, smooth_ce_loss, smooth_cl_loss = 0.0, 0.0, 0.0
    best_f1_dev, related_f1_test = 0.0, 0.0

    prompt_info = construct_prompt(processor.label_dict, processor.tokenizer) if args.use_label_semantics else None
    optimizer.zero_grad()
    if args.use_in_batch_cl:
        model_cl = Batch_CL(args, model, processor)
    else:
        model_cl = Momentum_CL(args, model, queue_size=args.queue_size)
        model_cl.set_queue_and_iter(train_features, processor)
    
    viterbi_decoder, crf_model = None, None
    if args.crf_strategy=='crf-inference':
        abstract_transitions = get_abstract_transitions(train_examples)
        viterbi_decoder = ViterbiDecoder(n_tag=args.num_labels+1, abstract_transitions=abstract_transitions, tau=args.crf_tau, device=args.device)
    elif args.crf_strategy=='crf' or args.crf_strategy=='crf-pa':
        crf_model = model.crf
    else:
        pass

    while global_step <= args.max_steps:
        step += 1
        model.train()
        try:
            ce_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            ce_batch = next(train_iter)
        inputs = {
            'input_ids':       ce_batch[0].to(args.device),
            'attention_mask':  ce_batch[1].to(args.device), 
            'label_ids':    torch.cat(ce_batch[5]).to(args.device),
            'start_list_list':   [start_list.to(args.device) for start_list in ce_batch[3]],
            'end_list_list':     [end_list.to(args.device) for end_list in ce_batch[4]],
            'prompt_input_ids': prompt_info[0].to(args.device) if args.use_label_semantics else None,
            'prompt_attention_mask': prompt_info[1].to(args.device) if args.use_label_semantics else None,
            'prompt_start_list': prompt_info[2] if args.use_label_semantics else None,
            'prompt_end_list': prompt_info[3] if args.use_label_semantics else None,
        }
        if args.fp_16:
            with autocast():
                outputs = model(**inputs)
                ce_loss, tok_embeds, ce_logits = outputs[0], outputs[2], outputs[4]
        else:
            outputs = model(**inputs)
            ce_loss, tok_embeds, ce_logits = outputs[0], outputs[2], outputs[4]
        smooth_ce_loss += ce_loss.item()/args.logging_steps

        cl_loss = torch.tensor([0.0], dtype=torch.float).to(args.device)
        if args.use_in_batch_cl:
            inputs = {
                'label_ids':    torch.cat(ce_batch[5]).to(args.device),
                'feature_embeds':  tok_embeds,   
            }
        else:
            inputs = {
                'input_ids':       ce_batch[0].to(args.device),
                'attention_mask':  ce_batch[1].to(args.device), 
                'label_ids':    torch.cat(ce_batch[5]).to(args.device),
                'start_list_list':   [start_list.to(args.device) for start_list in ce_batch[3]],
                'end_list_list':     [end_list.to(args.device) for end_list in ce_batch[4]],
                'query_embeds':  tok_embeds,   
            }
            if args.fp_16:
                with autocast():
                    cl_outputs = model_cl(**inputs)
                    cl_loss, cl_logits = cl_outputs[0], cl_outputs[1]
            else:
                cl_outputs = model_cl(**inputs)
                cl_loss, cl_logits = cl_outputs[0], cl_outputs[1]
            smooth_cl_loss += cl_loss.item()/args.logging_steps

        if args.fp_16:
            with autocast():
                loss = ce_loss + cl_loss
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
        else:
            loss = ce_loss + cl_loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
        smooth_loss += loss.item()/args.logging_steps
        
        if (step+1)%args.gradient_accumulation_steps==0:
            if args.fp_16:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:  
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if (global_step%args.logging_steps==0) and global_step>0 and (step+1)%args.gradient_accumulation_steps==0:
            logging.info("-----------------------global_step: {} -------------------------------- ".format(global_step))
            logging.info('lr: {}'.format(scheduler.get_last_lr()[0]))
            logging.info('smooth_loss: {}, smooth_ce_loss: {}, smooth_cl_loss:{}'.format(
                smooth_loss, smooth_ce_loss, smooth_cl_loss)
            )
            tb_writer.add_scalar('train_loss', smooth_loss, global_step)
            tb_writer.add_scalar('train_ce_loss', smooth_ce_loss, global_step)
            tb_writer.add_scalar('train_cl_loss', smooth_cl_loss, global_step)
            smooth_loss = .0
            smooth_ce_loss = .0
            smooth_cl_loss = .0

        if (global_step%args.eval_steps==0) and (step+1)%args.gradient_accumulation_steps==0 and global_step!=0:
            dev_micro_recall, dev_micro_precision, dev_micro_f1_knn, dev_gt_list, dev_pred_list, dev_knn_logits = \
                evaluate_knn(args, model, dev_features, train_features, processor, set_type="dev")
            if args.crf_strategy != 'no-crf':  
                dev_micro_recall, dev_micro_precision, dev_micro_f1_crf, dev_gt_list, dev_pred_list = \
                    evaluate_crf(args, dev_knn_logits, dev_features, set_type="dev", viterbi_decoder=viterbi_decoder, crf_model=crf_model)

            test_micro_recall, test_micro_precision, test_micro_f1_knn, test_gt_list, test_pred_list, test_knn_logits = \
                evaluate_knn(args, model, test_features, train_features, processor, set_type="test")
            if args.crf_strategy != 'no-crf':   
                test_micro_recall, test_micro_precision, test_micro_f1_crf, test_gt_list, test_pred_list = \
                    evaluate_crf(args, test_knn_logits, test_features, set_type="test", viterbi_decoder=viterbi_decoder, crf_model=crf_model) 

            if args.crf_strategy != 'no-crf':
                (dev_micro_f1, test_micro_f1) = (dev_micro_f1_crf, test_micro_f1_crf)
            else:
                (dev_micro_f1, test_micro_f1) = (dev_micro_f1_knn, test_micro_f1_knn) 

            eval_flag = (global_step>args.start_eval_steps) if (args.start_eval_steps<args.max_steps) else (global_step>args.max_steps//2)
            if eval_flag:
                if not os.path.exists(os.path.join(args.output_dir, 'checkpoint')):
                    os.makedirs(os.path.join(args.output_dir, 'checkpoint'))
                if dev_micro_f1 > best_f1_dev:
                    best_f1_dev = dev_micro_f1
                    related_f1_test = test_micro_f1
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(os.path.join(args.output_dir, 'checkpoint'))
                    torch.save(args, os.path.join(args.output_dir, 'checkpoint', 'training_args.bin'))
                    show_results(
                        dev_features, dev_gt_list, dev_pred_list, processor, os.path.join(args.output_dir, "dev_result.txt")
                    )
                    show_results(
                        test_features, test_gt_list, test_pred_list, processor, os.path.join(args.output_dir, "test_result.txt")
                    )

                logging.info('current best dev-f1 score: {}'.format(best_f1_dev))
                logging.info('current related test-f1 score: {}'.format(related_f1_test))
                tb_writer.add_scalar('dev_micro_f1', dev_micro_f1, global_step)
                tb_writer.add_scalar('dev_micro_r', dev_micro_recall, global_step)
                tb_writer.add_scalar('dev_micro_p', dev_micro_precision, global_step)

    res_record = {
        "dev_f1": best_f1_dev, 
        "test_f1": related_f1_test,
    }
    with open(os.path.join(args.output_dir, "res.json"), 'w') as f:
        json.dump(res_record, f)
    tb_writer.close()


def evaluate(args, model, processor, features, dataloader, prompt_info, set_type):
    model.eval()
    
    gt_list, pred_list = list(), list()
    tok_embeds, logits = list(), list()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids':       batch[0].to(args.device),
                'attention_mask':  batch[1].to(args.device), 
                'start_list_list':   [start_list.to(args.device) for start_list in batch[3]],
                'end_list_list':     [end_list.to(args.device) for end_list in batch[4]],
                'prompt_input_ids': prompt_info[0].to(args.device) if args.use_label_semantics else None,
                'prompt_attention_mask': prompt_info[1].to(args.device) if args.use_label_semantics else None,
                'prompt_start_list': prompt_info[2] if args.use_label_semantics else None,
                'prompt_end_list': prompt_info[3] if args.use_label_semantics else None,
            }

            outputs = model(**inputs)
            pred_labels, tok_embed, logit = outputs[1], outputs[2], outputs[4] 
            gt_res_list = batch[5]
            pred_res_list = cut_pred_res(pred_labels, gt_res_list)
        
            gt_list.extend([get_interval_res(gt_res, start_list) for gt_res, start_list in zip(gt_res_list, inputs["start_list_list"])])
            pred_list.extend([get_interval_res(pred_res, start_list) for pred_res, start_list in zip(pred_res_list, inputs["start_list_list"])])
            tok_embeds.append(tok_embed)
            logits.append(logit)
    tok_embeds = torch.cat(tok_embeds, dim=0)
    logits = torch.cat(logits, dim=0)
    micro_recall, micro_precision, micro_f1, gt_num, pred_num, correct_num= eval_score_seq_label(gt_list, pred_list)
    logger.info("({}) Classification\t\tmicro-P:{}\tmicro-R:{}\tmicro-F:{}\tPred num:{}\tGt num:{}\tCorrect num:{}".format(
        set_type, micro_precision, micro_recall, micro_f1, pred_num, gt_num, correct_num
    ))

    return micro_recall, micro_precision, micro_f1, gt_list, pred_list, tok_embeds, logits


def evaluate_knn(args, model, features, train_features, processor, set_type):
    model.eval()
    train_embeddings = compute_embeddings(model, train_features, processor, args, verbose=False)
    embeddings = torch.tensor(
        compute_embeddings(model, features, processor, args, verbose=False)
    ).float()
    gt_res_list, pred_labels = list(), list()

    for feature in features:
        if feature is None:
            continue
        gt_res_list.append(feature.label_list)  

    train_labels = list() 
    for feature in train_features:
        if feature is None: 
            continue
        train_labels.extend(feature.label_list) 
    train_labels = np.array(train_labels)
    train_label_index_dict = {label:np.where(train_labels==label)[0] for label in range(args.num_labels)}

    logits_nearest = torch.zeros((embeddings.size(0), args.num_labels)).float().to(args.device)
    with torch.no_grad():
        train_embeddings = torch.tensor(train_embeddings[None, :, :]).float().to(args.device)
        embeddings = embeddings[:, None, :].clone().detach().float().to(args.device)
        if args.use_tapnet:
            train_embeddings = F.normalize(train_embeddings@model.tapnet.M, p=2, dim=-1)
            embeddings = F.normalize(embeddings@model.tapnet.M, p=2, dim=-1)

        for start in range(0, embeddings.size(0), args.buffer_size):
            sub_embeddings = embeddings[start:(start+args.buffer_size)]
            dists = list()
            for embedding in sub_embeddings:
                dist = torch.sum((embedding - train_embeddings)**2, dim=-1)
                dists.append(dist)
            dists = torch.cat(dists, dim=0)
            
            for label in range(1, args.num_labels):
                try:
                    logits_nearest[start:(start+args.buffer_size), label] = -torch.min(dists[:, train_label_index_dict[label]], dim=-1).values
                except:
                    logits_nearest[start:(start+args.buffer_size), label] = -10000
            none_event_dists = dists[:, train_label_index_dict[0]]
            logits_nearest[start:(start+args.buffer_size), 0] = -torch.mean(none_event_dists.topk(k=1, dim=-1, largest=False).values, dim=-1)
    logits_nearest = logits_nearest/args.temperature
  
    start_list_list = [feature.start_list for feature in features if feature is not None]
    gt_list = [get_interval_res(gt_res, start_list) for gt_res, start_list in zip(gt_res_list, start_list_list)]
    pred_labels = logits_nearest.max(dim=-1).indices
    pred_res_list = cut_pred_res(pred_labels, gt_res_list) 
    pred_list = [get_interval_res(pred_res, start_list) for pred_res, start_list in zip(pred_res_list, start_list_list)]
    micro_recall, micro_precision, micro_f1, gt_num, pred_num, correct_num = eval_score_seq_label(gt_list, pred_list) 
    logger.info("({}) KNN nearest\t\tmicro-P:{}\tmicro-R:{}\tmicro-F:{}\tPred num:{}\tGt num:{}\tCorrect num:{}".format(
        set_type, micro_precision, micro_recall, micro_f1, pred_num, gt_num, correct_num
    ))

    return micro_recall, micro_precision, micro_f1, gt_list, pred_list, logits_nearest


def evaluate_crf(args, logits_knn, features, set_type, viterbi_decoder=None, crf_model=None):
    start_list_list = [feature.start_list for feature in features if feature is not None]
    gt_res_list = list()
    for feature in features:
        if feature is None: 
            continue
        gt_res_list.append(feature.label_list)  
    gt_list = [get_interval_res(gt_res, start_list) for gt_res, start_list in zip(gt_res_list, start_list_list)]

    pred_res_list = list()
    start_list_list = [feature.start_list for feature in features if feature is not None]
    if args.crf_strategy=='crf-inference':
        assert(viterbi_decoder is not None)
        logits_per_sent = cut_pred_res(logits_knn, gt_res_list)
        for logit in logits_per_sent:
            sent_len, n_label = logit.size()
            sent_probs = F.softmax(logit, dim=1)
            start_probs = torch.zeros(sent_len).to(args.device) + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len).detach().cpu().numpy()
            pred_res = vit_labels - 1   # SUBSTITUE THE START_ID
            pred_res_list.append(pred_res)
    elif args.crf_strategy=='crf' or args.crf_strategy=='crf-pa':
        assert(crf_model is not None)
        bs, sent_len = len(start_list_list), max([len(start_list) for start_list in start_list_list])
        attention_mask = torch.zeros((sent_len, bs)).to(logits_knn.device).float()
        for idx, start_list in enumerate(start_list_list):
            attention_mask[:len(start_list), idx] = 1.0

        flatten_emissions = torch.log(F.softmax(logits_knn, dim=-1))
        emissions = torch.zeros((attention_mask.size(0), attention_mask.size(1), logits_knn.size(-1))).to(logits_knn.device) - 100.0
        for idx, emission in enumerate(cut_pred_res(flatten_emissions, gt_res_list)):
            emissions[:len(emission), idx, :] = emission
        pred_res_list = crf_model.decode(emissions, mask=attention_mask.bool())
    else:
        pass
    pred_list = [get_interval_res(pred_res, start_list) for pred_res, start_list in zip(pred_res_list, start_list_list)] 
    micro_recall, micro_precision, micro_f1, gt_num, pred_num, correct_num = eval_score_seq_label(gt_list, pred_list)
    logger.info("({}) Ensemble(CRF)\t\tmicro-P:{}\tmicro-R:{}\tmicro-F:{}\tPred num:{}\tGt num:{}\tCorrect num:{}".format(
        set_type, micro_precision, micro_recall, micro_f1, pred_num, gt_num, correct_num
    )) 

    return micro_recall, micro_precision, micro_f1, gt_list, pred_list


def inference(args, model, processor, output_name=None):
    def flatten_res(gt_list, pred_list):
        flatten_gt_list, flatten_pred_list = list(), list()
        for gt, pred in zip(gt_list, pred_list):
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
                flatten_gt_list.append(res["gt"])
                flatten_pred_list.append(res["pred"])
        return flatten_gt_list, flatten_pred_list         

    viterbi_decoder, crf_model = None, None
    if args.crf_strategy == 'crf-inference':
        train_examples, _, _ = processor.generate_sent_dataloader('train')
        abstract_transitions = get_abstract_transitions(train_examples)
        viterbi_decoder = ViterbiDecoder(n_tag=args.num_labels+1, abstract_transitions=abstract_transitions, tau=args.crf_tau, device=args.device)
    elif args.crf_strategy == 'crf' or args.crf_strategy == 'crf-pa':
        crf_model = model.crf
    else:
        pass

    label_dict = {v:k for k,v in processor.label_dict.items()}
    res_report = defaultdict(OrderedDict)
    prompt_info = construct_prompt(processor.label_dict, processor.tokenizer) if args.use_label_semantics else None
    _, train_features, _ = processor.generate_sent_dataloader('train')
    _, dev_features, _ = processor.generate_sent_dataloader('dev')
    _, test_features, _ = processor.generate_sent_dataloader('test')

    dev_micro_recall, dev_micro_precision, dev_micro_f1_knn, dev_gt_list, dev_pred_list, dev_knn_logits = \
        evaluate_knn(args, model, dev_features, train_features, processor, set_type="dev")
    if args.crf_strategy != 'no-crf':   
        dev_micro_recall, dev_micro_precision, dev_micro_f1_crf, dev_gt_list, dev_pred_list = \
            evaluate_crf(args, dev_knn_logits, dev_features, set_type="dev", viterbi_decoder=viterbi_decoder, crf_model=crf_model) 
        dev_gt_list, dev_pred_list = flatten_res(dev_gt_list, dev_pred_list)
        dev_knn_report = classification_report(y_true=dev_gt_list, y_pred=dev_pred_list, labels=list(range(1, args.num_labels)), zero_division=0, output_dict=True)
        for label_id, res in dev_knn_report.items():
            try:
                event_type = label_dict[int(label_id)]
                res_report[event_type]['dev_knn'] = res
            except:
                res_report['overall']['dev_knn'] = res

    test_micro_recall, test_micro_precision, test_micro_f1_knn, test_gt_list, test_pred_list, test_knn_logits = \
        evaluate_knn(args, model, test_features, train_features, processor, set_type="test")
    if args.crf_strategy != 'no-crf':   
        test_micro_recall, test_micro_precision, test_micro_f1_crf, test_gt_list, test_pred_list = \
            evaluate_crf(args, test_knn_logits, test_features, set_type="test", viterbi_decoder=viterbi_decoder, crf_model=crf_model) 
        test_gt_list, test_pred_list = flatten_res(test_gt_list, test_pred_list)
        test_knn_report = classification_report(y_true=test_gt_list, y_pred=test_pred_list, labels=list(range(1, args.num_labels)), zero_division=0, output_dict=True)
        for label_id, res in test_knn_report.items():
            try:
                event_type = label_dict[int(label_id)]
                res_report[event_type]['test_knn'] = res
            except:
                res_report['overall']['test_knn'] = res

    train_trigger_dict = processor.generate_trigger_dict(train_features, set_type="train", count=True)
    dev_trigger_dict = processor.generate_trigger_dict(dev_features, set_type="dev", count=True)
    test_trigger_dict = processor.generate_trigger_dict(test_features, set_type="test", count=True)
    for label_id in label_dict:
        if label_id==0:
            continue
        event_type = label_dict[int(label_id)]
        res_report[event_type]['train_trigger'] = train_trigger_dict[label_id] if label_id in train_trigger_dict else list()
        res_report[event_type]['dev_trigger'] = dev_trigger_dict[label_id] if label_id in dev_trigger_dict else list()
        res_report[event_type]['test_trigger'] = test_trigger_dict[label_id] if label_id in test_trigger_dict else list()

    output_path = os.path.join(args.output_dir, "res_report.json") if output_name is None else os.path.join(args.output_dir, "res_report_{}.json".format(output_name))
    with open(output_path, 'w') as f:
        json.dump(res_report, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./data/ACE05_processed/train.json", type=str)
    parser.add_argument("--dev_file", default="./data/ACE05_processed/dev.json", type=str)
    parser.add_argument("--test_file", default="./data/ACE05_processed/test.json", type=str)
    parser.add_argument("--output_dir", default='./outputs_exp', type=str)
    parser.add_argument("--dataset_type", default='ACE', type=str)
    parser.add_argument("--model_type", default='roberta', type=str)
    parser.add_argument("--config_name", default=None, type=str)
    parser.add_argument("--tokenizer_name", default=None, type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--label_dict_path", type=str, default=None)
    
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--max_seq_length", default=168, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument('--logging_steps', default=100, type=int)
    parser.add_argument('--eval_steps', default=500, type=int)
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument("--start_eval_steps", default=0, type=int)
    parser.add_argument('--tenured_num', default=64, type=int)
    parser.add_argument('--queue_size', default=2048, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--alpha_momentum', default=0.99, type=float)
    parser.add_argument('--projection_dimension', default=128, type=int)
    parser.add_argument('--dist_func', default='dot-product', choices=['dot-product', 'euclidean', 'KL'], type=str)
    parser.add_argument('--crf_strategy', default='no-crf', choices=['no-crf', 'crf-inference', 'crf', 'crf-pa'])
    parser.add_argument('--crf_tau', default=0.05, type=float)

    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--drop_last', action='store_true', default=False)
    parser.add_argument('--use_in_batch_cl', action='store_true', default=False)
    parser.add_argument('--use_label_semantics', action='store_true', default=False)
    parser.add_argument('--label_score_enhanced', action='store_true', default=False)
    parser.add_argument('--label_feature_enhanced', action='store_true', default=False)
    parser.add_argument('--use_tapnet', action='store_true', default=False)
    parser.add_argument('--use_normalize', action='store_true', default=False)
    parser.add_argument('--drop_none_event', action='store_true', default=False)
    parser.add_argument("--fp_16", default=False, action="store_true")
    args = parser.parse_args()
    args.num_labels = len(read_json(args.label_dict_path))
    set_seed(args)

    if args.inference:
        ckpt_dir = os.path.join(args.output_dir, 'checkpoint')
        args.tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path
        args.model_name_or_path = ckpt_dir
        args.config_name = ckpt_dir
        logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
        )
        logger.info("Training/evaluation parameters %s", args)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
        )
        logger.info("Training/evaluation parameters %s", args)
        with open(os.path.join(args.output_dir, 'commandline_args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
  
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.num_labels
    config.temperature = args.temperature
    config.use_label_semantics = args.use_label_semantics
    config.label_score_enhanced = args.label_score_enhanced
    config.label_feature_enhanced = args.label_feature_enhanced
    config.use_normalize = args.use_normalize
    config.crf_strategy = args.crf_strategy
    config.use_tapnet = args.use_tapnet
    config.dist_func = args.dist_func

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=False)
    processor = EventDetectionProcessor(args, tokenizer)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, ignore_mismatched_sizes=True).to(args.device)
    model.resize_token_embeddings(len(processor.tokenizer))

    if not args.inference:
        train(args, model, processor)
        ckpt_dir = os.path.join(args.output_dir, 'checkpoint')
        logger.info("Loading checkpoint from previous checkpoint: {}".format(ckpt_dir))
        # Inference after training
        config = config_class.from_pretrained(ckpt_dir)
        model = model_class.from_pretrained(ckpt_dir, config=config).to(args.device)  
    inference(args, model, processor)
            

if __name__ == "__main__":
    main()