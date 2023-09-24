import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import construct_prompt, _loss_kl


class Batch_CL(nn.Module):
    def __init__(self, args, model, processor):
        super().__init__()
        self.args = args
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.embedding_dim = args.projection_dimension

        if self.args.dist_func=="KL":
            self.output_embedder_mu = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_size,
                        self.embedding_dim)
            ).to(self.args.device)

            self.output_embedder_sigma = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_size,
                        self.embedding_dim)
            ).to(self.args.device)

        self.prompt_info = [a.to(self.args.device) if isinstance(a, torch.Tensor) else a for a in construct_prompt(processor.label_dict, processor.tokenizer)]

    def forward(
        self,
        feature_embeds=None,
        label_ids=None,
        th_masks=None
    ):
        
        if self.args.use_normalize:
            feature_embeds = F.normalize(feature_embeds, dim=-1)
        if self.args.drop_none_event:
            feature_embeds = feature_embeds[label_ids>0]
            label_ids = label_ids[label_ids>0]
        
        if self.args.dist_func=="KL":
            mu_embeds = self.output_embedder_mu(feature_embeds)
            sigma_embeds = F.elu(self.output_embedder_sigma(feature_embeds)) + 1 + 1e-14
        else:
            if self.args.use_normalize:
                feature_embeds = F.normalize(feature_embeds, dim=-1)
        
        if self.args.use_tapnet:
            prototypes = torch.zeros((self.args.num_labels, feature_embeds.size(-1)), requires_grad=True).to(self.args.device).float()
            for label_id in range(self.args.num_labels):
                feature_embeds_this_label = feature_embeds[label_ids==label_id]
                if len(feature_embeds_this_label)>0:
                    prototypes[label_id] = torch.mean(feature_embeds_this_label, dim=0)

            prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list = self.prompt_info
            prompt_embeds = self.model.compute_prompt_embeddings(prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list) if self.args.label_feature_enhanced else None
            M = self.model.tapnet(
                prototypes, 
                use_normalize=self.args.use_normalize, 
                prompt_embeds=prompt_embeds, 
                label_feature_enhanced=self.args.label_feature_enhanced, 
                use_prototype_reference=self.args.label_feature_enhanced,
            )
            if self.args.use_normalize:
                feature_embeds = F.normalize(feature_embeds@M, p=2, dim=-1)

        numerator_matrix, denominator_matrix = self.generate_label_mask(label_ids)
        if self.args.dist_func=="KL":
            logits = -_loss_kl(mu_embeds, sigma_embeds, mu_embeds, sigma_embeds, embed_dimension=self.embedding_dim)
        elif self.args.dist_func=="euclidean":
            logits = list()
            for feature_embed in feature_embeds:
                logit = -torch.sum((feature_embed - feature_embeds)**2, dim=-1)
                logits.append(logit.unsqueeze(0))
            logits = torch.cat(logits, dim=0)
            logits /= self.args.temperature
        else:
            logits = feature_embeds@feature_embeds.T
            logits /= self.args.temperature
        logits = logits.exp()

        denominator_matrix = denominator_matrix[torch.sum(numerator_matrix, dim=-1)>0]
        logits = logits[torch.sum(numerator_matrix, dim=-1)>0]
        numerator_matrix = numerator_matrix[torch.sum(numerator_matrix, dim=-1)>0]

        Z = torch.sum(logits*denominator_matrix, dim=-1)
        loss = -torch.mean(
            torch.sum(torch.log(logits/Z[:, None])*numerator_matrix, dim=-1)/torch.sum(numerator_matrix, dim=-1), dim=-1
        )
        return loss, None
        
        
    def generate_label_mask(self, label_ids):
        L = label_ids.size(0)
        numerator_label_mask = (label_ids[:, None]==label_ids[None, :]).float()
        denominator_label_mask = torch.ones_like(numerator_label_mask)
        numerator_label_mask[torch.arange(L), torch.arange(L)] = 0.0
        denominator_label_mask[torch.arange(L), torch.arange(L)] = 0.0
        return numerator_label_mask, denominator_label_mask