"""
This code is slightly modified from https://github.com/congxin95/PA-CRF
"""


import torch
import torch.nn as nn
from torchcrf import CRF


class PA_CRF(CRF):
    def __init__(self, num_tags, feature_size):
        super().__init__(num_tags)
        self.feature_size = feature_size
        
        # self attention
        self.Wk = nn.Linear(self.feature_size, self.feature_size)
        self.Wq = nn.Linear(self.feature_size, self.feature_size)
        self.Wv = nn.Linear(self.feature_size, self.feature_size)
    
        # crf score
        self.W_start_mean = nn.Linear(self.feature_size, 1)
        self.W_start_log_var = nn.Linear(self.feature_size, 1)
            
        self.W_end_mean = nn.Linear(self.feature_size, 1)
        self.W_end_log_var = nn.Linear(self.feature_size, 1)
        
        self.W_trans_mean = nn.Linear(self.feature_size * 2, 1)
        self.W_trans_log_var = nn.Linear(self.feature_size * 2, 1)


    def set_transitions(self, start_transitions, end_transitions, transitions):
        self.start_transitions = nn.Parameter(start_transitions)
        self.end_transitions = nn.Parameter(end_transitions)
        self.transitions = nn.Parameter(transitions)

    
    def proto_interaction(self, prototype):
         # self attention
        K = self.Wk(prototype)  # L, feature_size
        Q = self.Wq(prototype)  # L, feature_size
        V = self.Wv(prototype)  # L, feature_size
        
        att_score = torch.matmul(K, Q.transpose(-1, -2))                # L, L
        att_score /= torch.sqrt(torch.tensor(self.feature_size).to(K))  # L, L
        att_score = att_score.softmax(-1)                               # L, L
        
        prototype = torch.matmul(att_score, V)  # L, feature_size
        return prototype


    def compute_trans_score(self, prototype):
        label_num, _ = prototype.shape
        left_prototype = prototype.unsqueeze(0).expand(label_num, label_num, -1)
        right_prototype = prototype.unsqueeze(1).expand(label_num, label_num, -1)
        cat_prototype = torch.cat([left_prototype, right_prototype], dim=-1)
        
        trans_mean = self.W_trans_mean(cat_prototype).squeeze(-1)             # B, 2*N+1, feature_size        
        trans_log_var = self.W_trans_log_var(cat_prototype).squeeze(-1)
        
        trans_score = self.sampling(trans_mean, trans_log_var)
        
        return trans_score


    def generate_transition_score(self, prototype):
        # calculate crf score
        start_mean = self.W_start_mean(prototype).squeeze(-1)   # L,
        start_log_var = self.W_start_log_var(prototype).squeeze(-1)   # L, 
        start_score = self.sampling(start_mean, start_log_var)   # L, 
        
        end_mean = self.W_end_mean(prototype).squeeze(-1)       # L,
        end_log_var = self.W_end_log_var(prototype).squeeze(-1)       # L,
        end_score = self.sampling(end_mean, end_log_var)       # L,
        
        # reparameterize
        trans_score = self.compute_trans_score(prototype)
        
        return start_score, end_score, trans_score
    

    def sampling(self, mean, logvar):
        epsilon = torch.randn(*mean.shape).to(mean.device)
        samples = mean + torch.exp(0.5 * logvar) * epsilon
        return samples


    def get_transition_score(self, prototype):
        # self attention
        prototype = self.proto_interaction(prototype)
        prototype = prototype.relu()
        
        # calculate crf score        
        start_score, end_score, trans_score = self.generate_transition_score(prototype)
        
        return start_score, end_score, trans_score