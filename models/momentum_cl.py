import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from tqdm import tqdm
from copy import deepcopy
from utils import get_label_mask, construct_prompt, _loss_kl


logger = logging.getLogger(__name__)


class EventQueue:
    @torch.no_grad()
    def __init__(self, args, processor, sent_features, queue_size, dim=768):
        self.args = args
        self.ptr_tenured = 0
        self.dim = dim
        self.eval_batch_size = args.eval_batch_size
        self.queue_size = queue_size

        self.tenured_features, self.tenured_dataloader = processor.generate_tenure_features(sent_features)
        if self.args.use_normalize and self.args.dist_func!='KL':
            self.tenured_queue = F.normalize(torch.randn(self.dim, len(self.tenured_features)), dim=0) 
        self.tenured_label_ids = torch.tensor([feature.label_id for feature in self.tenured_features]).to(self.args.device)

        self.non_tenured_queue = None
        self.non_tenured_label_ids = None
        self.prompt_info = [a.to(self.args.device) if isinstance(a, torch.Tensor) else a for a in construct_prompt(processor.label_dict, processor.tokenizer)]

    @property
    def embeds(self):
        return torch.cat([self.tenured_queue, self.non_tenured_queue], dim=1) 

    @property
    def label_ids(self):
        return torch.cat([self.tenured_label_ids, self.non_tenured_label_ids])

    @property
    def queue_prototypes(self):
        prototypes = torch.zeros((self.args.num_labels, self.dim), requires_grad=True).to(self.args.device).float()
        for id in range(self.args.num_labels):
            label_idx = self.label_ids[self.label_ids==id]
            if len(label_idx)>0:
                prototypes[id, :] = torch.mean(self.embeds[:, label_idx], dim=-1)
        return F.normalize(prototypes, p=2, dim=-1) if (self.args.use_normalize and self.args.dist_func!='KL') \
            else prototypes


    @torch.no_grad()
    def init_queue(self, model):
        logger.info("Start queue initialization")
        tok_embed_list = list()
        for batch in tqdm(self.tenured_dataloader):
            inputs = {
                'input_ids':       batch[0].to(self.args.device),
                'attention_mask':  batch[1].to(self.args.device), 
                'start_list_list':   [torch.tensor([start]).to(self.args.device) for start in batch[4]],
                'end_list_list':     [torch.tensor([start]).to(self.args.device) for start in batch[5]],
                'compute_features_only': True,
            }
            tok_embeds = model(**inputs)
            tok_embed_list.append(tok_embeds.clone().detach())
        tok_embed_list = torch.cat(tok_embed_list, dim=0)
        if self.args.use_normalize and self.args.dist_func!='KL':
            tok_embed_list = F.normalize(tok_embed_list, dim=1)
        assert(tok_embed_list.size(0)==len(self.tenured_features))
        self.tenured_queue = tok_embed_list.T
        logger.info("Finish queue initialization")


    @torch.no_grad()
    def update_tenured_queue(self, model):
        tenured_input_ids = torch.tensor(
            [feature.enc_input_ids for feature in self.tenured_features[self.ptr_tenured:(self.ptr_tenured+self.args.tenured_num)]]
        ).to(self.args.device)
        tenured_mask_ids = torch.tensor(
            [feature.enc_mask_ids for feature in self.tenured_features[self.ptr_tenured:(self.ptr_tenured+self.args.tenured_num)]]
        ).to(self.args.device)
        tenured_tok_start = [torch.tensor([feature.tok_start]).to(self.args.device) \
            for feature in self.tenured_features[self.ptr_tenured:(self.ptr_tenured+self.args.tenured_num)]
        ]
        tenured_tok_end = [torch.tensor([feature.tok_end]).to(self.args.device) \
            for feature in self.tenured_features[self.ptr_tenured:(self.ptr_tenured+self.args.tenured_num)]
        ]

        inputs = {
            'input_ids':       tenured_input_ids,
            'attention_mask':  tenured_mask_ids, 
            'start_list_list':   tenured_tok_start,
            'end_list_list':     tenured_tok_end,
            'compute_features_only': True,
        }
        tok_embeds = model(**inputs)
        tok_embeds = tok_embeds.clone().detach()
        if self.args.use_normalize and self.args.dist_func!='KL':
            tok_embeds = F.normalize(tok_embeds, dim=-1).T
        else:
            tok_embeds = tok_embeds.T
        self.tenured_queue[:, self.ptr_tenured:(self.ptr_tenured+self.args.tenured_num)] = tok_embeds
        if self.ptr_tenured + self.args.tenured_num >= len(self.tenured_features):
            self.ptr_tenured = 0
        else:
            self.ptr_tenured += self.args.tenured_num
        return tok_embeds


    @torch.no_grad()
    def update_nontenured_queue(self, model, input_ids, attention_mask, label_ids, start_list_list, end_list_list):
        inputs = {
            'input_ids':       input_ids,
            'attention_mask':  attention_mask, 
            'start_list_list':   start_list_list,
            'end_list_list':     end_list_list,
            'compute_features_only': True,
        }
        non_tenured_embeds = model(**inputs)
        if self.args.use_normalize and self.args.dist_func!='KL':
            non_tenured_embeds = F.normalize(non_tenured_embeds, p=2, dim=-1).T
        else:
            non_tenured_embeds = non_tenured_embeds.T
        assert(non_tenured_embeds.size(1)==label_ids.size(0))
        if self.args.drop_none_event:
            non_tenured_embeds = non_tenured_embeds[:, label_ids.bool()]
            label_ids = label_ids[label_ids.bool()]

        if self.non_tenured_queue is None:
            self.non_tenured_queue = non_tenured_embeds
            self.non_tenured_label_ids = label_ids
        else:
            self.non_tenured_queue = torch.cat([self.non_tenured_queue, non_tenured_embeds], dim=1)
            self.non_tenured_label_ids = torch.cat([self.non_tenured_label_ids, label_ids])
            if self.non_tenured_queue.size(1)>self.queue_size:
                self.non_tenured_queue = self.non_tenured_queue[:, -self.queue_size:]
                self.non_tenured_label_ids = self.non_tenured_label_ids[-self.queue_size:]


class Momentum_CL(nn.Module):
    def __init__(self, args, model, dim=768, queue_size=2048):
        """
        dim: feature dimension (default: 768)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super().__init__()

        self.queue_size = queue_size
        self.dim = dim
        self.args = args
        self.embedding_dim = args.projection_dimension

        # create the encoders
        self.encoder_q = model
        self.encoder_k = deepcopy(model)

        if self.args.dist_func=="KL":
            self.output_embedder_mu = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.dim,
                        self.embedding_dim)
            ).to(self.args.device)

            self.output_embedder_sigma = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.dim,
                        self.embedding_dim)
            ).to(self.args.device)

        backbone_q = self.encoder_q.roberta if self.args.model_type=='roberta' else self.encoder_q.model
        backbone_k = self.encoder_k.roberta if self.args.model_type=='roberta' else self.encoder_k.model
        for param_q, param_k in zip(backbone_q.parameters(), backbone_k.parameters()):
            assert(torch.equal(param_k.data, param_q.data))
            param_k.requires_grad = False  # not update by gradient

        
    def set_queue_and_iter(self, features, processor):    
        self.event_queue = EventQueue(self.args, processor, features, self.queue_size, dim=self.dim)
        self.event_queue.init_queue(self.encoder_k)


    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        backbone_q = self.encoder_q.roberta if self.args.model_type=='roberta' else self.encoder_q.model
        backbone_k = self.encoder_k.roberta if self.args.model_type=='roberta' else self.encoder_k.model
        for param_q, param_k in zip(backbone_q.parameters(), backbone_k.parameters()):
            param_k.data = param_k.data * self.args.alpha_momentum + param_q.data * (1. - self.args.alpha_momentum)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        label_ids=None,
        start_list_list=None,
        end_list_list=None,
        query_embeds=None,
    ):
        if self.args.use_normalize and self.args.dist_func!='KL':
            query_embeds = F.normalize(query_embeds, dim=-1)
        assert(query_embeds.requires_grad)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self.event_queue.update_tenured_queue(self.encoder_k)
            self.event_queue.update_nontenured_queue(
                model=self.encoder_k,
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_ids=label_ids,
                start_list_list=start_list_list,
                end_list_list=end_list_list,
            )
            key_embeds = self.event_queue.embeds

        if self.args.drop_none_event:
            numerator_matrix, denominator_matrix = get_label_mask(label_ids[label_ids.bool()], self.event_queue.label_ids)
            query_embeds = query_embeds[label_ids.bool()]
        else:
            numerator_matrix, denominator_matrix = get_label_mask(label_ids, self.event_queue.label_ids)
        
        if self.args.use_tapnet:
            prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list = self.event_queue.prompt_info
            if self.args.use_normalize and self.args.dist_func!='KL':
                prompt_embeds = F.normalize(
                    self.encoder_q.compute_prompt_embeddings(prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list),
                    p=2, dim=-1
                )
            else:
                prompt_embeds = self.encoder_q.compute_prompt_embeddings(prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list)
            prototypes = self.event_queue.queue_prototypes
            M = self.encoder_q.tapnet(
                prototypes, 
                use_normalize=self.args.use_normalize, 
                prompt_embeds=prompt_embeds, 
                label_feature_enhanced=self.args.label_feature_enhanced, 
                use_prototype_reference=self.args.label_feature_enhanced
            )
            if self.args.use_normalize and self.args.dist_func!='KL':
                query_embeds = F.normalize(query_embeds@M, p=2, dim=-1)
                key_embeds = F.normalize(key_embeds.T@M, p=2, dim=-1).T.detach()  
            else:
                query_embeds = query_embeds@M
                key_embeds = (key_embeds.T@M).T.detach()
            key_embeds.requires_grad = False
        
        if self.args.dist_func=='euclidean':
            logits = (-torch.sum((query_embeds.unsqueeze(1) - key_embeds.T.unsqueeze(0))**2, dim=-1))/self.args.temperature
        elif self.args.dist_func=='KL':
            query_mu = self.output_embedder_mu(query_embeds)
            query_sigma = F.elu(self.output_embedder_sigma(query_embeds)) + 1 + 1e-14
            key_mu = self.output_embedder_mu(key_embeds.T)
            key_sigma = F.elu(self.output_embedder_sigma(key_embeds.T)) + 1 + 1e-14
            logits = -_loss_kl(query_mu, query_sigma, key_mu, key_sigma, embed_dimension=self.embedding_dim)
        else:
            logits = (query_embeds@key_embeds)/self.args.temperature
        logits = logits.exp()
        Z = torch.sum(logits*denominator_matrix, dim=-1)
        
        logits_ = logits[torch.sum(numerator_matrix, dim=-1)>0]
        Z_ = Z[torch.sum(numerator_matrix, dim=-1)>0]
        numerator_matrix_ = numerator_matrix[torch.sum(numerator_matrix, dim=-1)>0]
        loss = -torch.mean(
            torch.sum(torch.log(logits_/Z_[:, None])*numerator_matrix_, dim=-1)/torch.sum(numerator_matrix_, dim=-1), dim=-1
        )

        return loss, logits