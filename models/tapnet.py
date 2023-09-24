import torch
import torch.nn as nn
import torch.nn.functional as F


class TapNet(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.reference_points = nn.Parameter(
            nn.init.xavier_normal_(torch.randn((self.num_labels, self.hidden_size))), requires_grad=True
        )

    
    def forward(self, prototypes, use_normalize, label_feature_enhanced=False, use_prototype_reference=False, prompt_embeds=None):
        if use_normalize:
            self.reference_points.data = F.normalize(self.reference_points.data, p=2, dim=-1)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            if prompt_embeds is not None:
                prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1)

        phis = self.reference_points
        if label_feature_enhanced:
            phis = phis + prompt_embeds 
        if use_prototype_reference:
            phis = phis + prototypes

        ref_hat_points = list()
        for i in range(self.num_labels):
            index = torch.ones((self.num_labels, )).to(prototypes.device).bool()
            index[i] = False
            ref_hat = phis[i] - torch.mean(prototypes[index], dim=0)
            ref_hat_points.append(ref_hat.unsqueeze(0))
        ref_hat_points = torch.cat(ref_hat_points, dim=0)

        errors = F.normalize(ref_hat_points, p=2, dim=-1) - F.normalize(prototypes, p=2, dim=-1)
        _, _, V = torch.svd(errors, some=False)
        self.M = V[:, self.num_labels:]

        return self.M