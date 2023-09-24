import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaPreTrainedModel

from torchcrf import CRF
from crf.pa_crf import PA_CRF
from models.tapnet import TapNet
from utils import cut_pred_res

logger = logging.getLogger(__name__)


class EventDetectionModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.temperature = config.temperature
        self.use_label_semantics = config.use_label_semantics
        self.label_feature_enhanced = config.label_feature_enhanced
        self.label_score_enhanced = config.label_score_enhanced
        self.use_normalize = config.use_normalize
        self.crf_strategy = config.crf_strategy
        self.dist_func = config.dist_func

        self.roberta = RobertaModel(config)
        if not self.use_label_semantics:
            self.fc_layer = nn.Linear(config.hidden_size, config.num_labels)
            self.roberta._init_weights(self.fc_layer)        
            if self.use_normalize:
                self.fc_layer.weight.data = F.normalize(self.fc_layer.weight.data, p=2, dim=-1)
        else:
            self.fc_layer = None
        if self.crf_strategy == 'crf':
            self.crf = CRF(self.num_labels)
        elif self.crf_strategy == 'crf-pa':
            self.crf = PA_CRF(self.num_labels, self.hidden_size)
        else:
            pass
        self.loss_fn = nn.CrossEntropyLoss()

        if self.config.use_tapnet:
            self.tapnet = TapNet(self.hidden_size, self.num_labels)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        label_ids=None,
        start_list_list=None,
        end_list_list=None,
        prompt_input_ids=None,
        prompt_attention_mask=None, 
        prompt_start_list=None,
        prompt_end_list=None,
        compute_features_only=False,
        output_prob=False,
        prototypes=None,
    ):

        sequence_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )[0]
        tok_embeds = list()
        for (seq, start_list, end_list) in zip(sequence_output, start_list_list, end_list_list):
            for (start, end) in zip(start_list, end_list):
                embed = torch.mean(seq[start:end], dim=0, keepdim=True)
                tok_embeds.append(embed)
        tok_embeds = torch.cat(tok_embeds, dim=0)
        if self.use_normalize:
            tok_embeds = F.normalize(tok_embeds, p=2, dim=-1)
        assert(tok_embeds.dim()==2 and tok_embeds.size(1)==self.hidden_size)

        if prototypes is not None:
            prompt_embeds = self.compute_prompt_embeddings(prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list) if self.label_feature_enhanced else None
            if self.config.use_tapnet:
                M = self.tapnet(
                    prototypes, 
                    use_normalize=self.use_normalize, 
                    prompt_embeds=prompt_embeds, 
                    label_feature_enhanced=self.label_feature_enhanced, 
                    use_prototype_reference=self.label_feature_enhanced    # We only consider such setting: (1) Label-enhanced + prototype reference. (2) None
                )
                tok_embeds = tok_embeds@M
                prototypes = prototypes@M
            else:
                if self.label_feature_enhanced:
                    prototypes = 0.5*(prototypes + prompt_embeds) 
            if self.use_normalize:
                tok_embeds = F.normalize(tok_embeds, p=2, dim=-1)
                prototypes = F.normalize(prototypes, p=2, dim=-1)
                prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1) if self.label_feature_enhanced else None

        if compute_features_only:
            return tok_embeds
        else:
            total_loss, pred_labels, pred_probs = None, None, None
            if prototypes is not None:
                logits = F.normalize(tok_embeds, p=2, dim=-1)@F.normalize(prototypes, p=2, dim=-1).T
            elif not self.use_label_semantics:
                logits = self.fc_layer(tok_embeds)
            else:
                prompt_embeds = self.compute_prompt_embeddings(prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list)
                if self.use_normalize:
                    prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1)
                assert(prompt_embeds.dim()==2 and prompt_embeds.size(1)==self.hidden_size and prompt_embeds.size(0)==self.num_labels)
                # logits = F.normalize(tok_embeds, p=2, dim=-1)@F.normalize(prompt_embeds, p=2, dim=-1).T
                logits = tok_embeds@prompt_embeds.T
            
            logits /= self.temperature
            if label_ids is not None:
                if self.crf_strategy=='crf':
                    total_loss = self.loss_crf(logits, label_ids, start_list_list)
                elif self.crf_strategy=='crf-pa':
                    if prototypes is None:
                        prototypes = prompt_embeds
                    start_score, end_score, trans_score = self.crf.get_transition_score(prototypes)
                    self.crf.set_transitions(start_score, end_score, trans_score)
                    total_loss = self.loss_crf(logits, label_ids, start_list_list)
                else:
                    total_loss = self.loss_fn(logits, label_ids)
            else:
                if output_prob:
                    pred_probs = F.softmax(logits, dim=-1)
                pred_labels = logits.max(dim=-1).indices
            return total_loss, pred_labels, tok_embeds, pred_probs, logits


    def forward_proto(
        self,
        prototypes,
        query_embeds,
        query_labels,
        start_list_list,
        prompt_input_ids=None,
        prompt_attention_mask=None, 
        prompt_start_list=None,
        prompt_end_list=None,
    ):
        prompt_embeds = self.compute_prompt_embeddings(
            prompt_input_ids, 
            prompt_attention_mask, 
            prompt_start_list, prompt_end_list
        ) if (self.label_feature_enhanced or self.label_score_enhanced) else None
        if self.config.use_tapnet:
            M = self.tapnet(
                prototypes, 
                use_normalize=self.use_normalize, 
                prompt_embeds=prompt_embeds, 
                label_feature_enhanced=self.label_feature_enhanced, 
                use_prototype_reference=self.label_feature_enhanced,
            )
            query_embeds = query_embeds@M
            prototypes = prototypes@M
            if self.use_normalize:
                query_embeds = F.normalize(query_embeds, p=2, dim=-1)
                prototypes = F.normalize(prototypes, p=2, dim=-1)
        else:
            if self.label_feature_enhanced:
                prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1) if self.use_normalize else prompt_embeds
                prototypes = F.normalize(prototypes + prompt_embeds, p=2, dim=-1) if self.use_normalize else 0.5*(prototypes+prompt_embeds)
            elif self.label_score_enhanced:
                prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1) if self.use_normalize else prompt_embeds
                prototypes = F.normalize(prototypes, p=2, dim=-1) if self.use_normalize else prototypes
            else:
                prototypes = F.normalize(prototypes, p=2, dim=-1) if self.use_normalize else prototypes

        logits = list()
        for query_embed in query_embeds:
            assert self.dist_func in ['euclidean', 'dot-product']
            if self.dist_func=='euclidean':
                logit = - torch.sqrt(torch.sum((query_embed.unsqueeze(0)-prototypes)**2, dim=-1).unsqueeze(0))
            else:
                logit = F.normalize(query_embed.unsqueeze(0), p=2, dim=-1)@F.normalize(prototypes, p=2, dim=-1).T
            logits.append(logit)
        logits = torch.cat(logits, dim=0)
        if self.label_score_enhanced:
            logits = (logits+query_embeds@prompt_embeds.T)/2
        logits /= self.config.temperature
        
        if self.crf_strategy=='crf':
            loss = self.loss_crf(logits, query_labels, start_list_list)
        elif self.crf_strategy=='crf-pa':
            start_score, end_score, trans_score = self.crf.get_transition_score(prototypes)
            self.crf.set_transitions(start_score, end_score, trans_score)
            loss = self.loss_crf(logits, query_labels, start_list_list)
        else:
            loss = F.cross_entropy(logits, query_labels, reduction="mean")
        return loss, logits

    
    def compute_prompt_embeddings(
        self,
        prompt_input_ids=None,
        prompt_attention_mask=None, 
        prompt_start_list=None,
        prompt_end_list=None,
    ):
        prompt_output = self.roberta(
            prompt_input_ids,
            attention_mask=prompt_attention_mask,
        )[0]
        prompt_embeds = list()
        for (seq, start, end) in zip(prompt_output, prompt_start_list, prompt_end_list):
            embed = torch.mean(seq[start:end], dim=0, keepdim=True)
            prompt_embeds.append(embed)
        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        return prompt_embeds

    
    def loss_crf(self, logits, label_ids, start_list_list):
        bs, sent_len = len(start_list_list), max([len(start_list) for start_list in start_list_list])
        attention_mask = torch.zeros((sent_len, bs)).to(logits.device).float()
        for idx, start_list in enumerate(start_list_list):
            attention_mask[:len(start_list), idx] = 1.0

        flatten_emissions = torch.log(F.softmax(logits, dim=-1))
        emissions = torch.zeros((attention_mask.size(0), attention_mask.size(1), logits.size(-1))).to(logits.device) - 100.0
        for idx, emission in enumerate(cut_pred_res(flatten_emissions, start_list_list)):
            emissions[:len(emission), idx, :] = emission

        tags = torch.zeros_like(attention_mask).long()
        for idx, label_id in enumerate(cut_pred_res(label_ids, start_list_list)):
            tags[:len(label_id), idx] = label_id
        
        loss = -self.crf(emissions, tags, mask=attention_mask.bool(), reduction='token_mean')
        return loss