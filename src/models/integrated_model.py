import torch
import torch.nn as nn
from src.models.lstm import TemporalBiLSTM
from src.models.fuzzy import NeuroFuzzyLayer

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(key_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key_value):
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        attention_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights, v)
        return self.out_proj(context)

class IntegratedCryptoModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_rules, dropout=0.2, sentiment_input_dim=5):
        super(IntegratedCryptoModel, self).__init__()
        self.lstm = TemporalBiLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, dropout)

        lstm_output_dim = lstm_hidden_size * 2

        self.sentiment_proj = nn.Linear(sentiment_input_dim, lstm_output_dim)
        self.attention = CrossAttention(query_dim=lstm_output_dim, key_dim=lstm_output_dim, hidden_dim=lstm_output_dim)
        
        fuzzy_input_size  = lstm_output_dim
        self.fuzzy        = NeuroFuzzyLayer(fuzzy_input_size, num_rules)
        self.combined_norm = nn.LayerNorm(lstm_output_dim)
        self.direct_head  = nn.Linear(lstm_output_dim, 1)

    def forward(self, x_seq, sentiment_scalar):
        lstm_sequence_out = self.lstm(x_seq)
        sentiment_query   = self.sentiment_proj(sentiment_scalar).unsqueeze(1)
        attended_context  = self.attention(query=sentiment_query, key_value=lstm_sequence_out)
        combined_input    = self.combined_norm(attended_context.squeeze(1))

        return self.fuzzy(combined_input) + self.direct_head(combined_input)