import torch
import torch.nn as nn
from src.models.lstm import TemporalBiLSTM
from src.models.fuzzy import NeuroFuzzyLayer

class IntegratedCryptoModel(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_rules, dropout=0.2):
        super(IntegratedCryptoModel, self).__init__()
        self.lstm = TemporalBiLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, dropout)
        fuzzy_input_size = (lstm_hidden_size * 2) + 1
        self.fuzzy = NeuroFuzzyLayer(fuzzy_input_size, num_rules)

    def forward(self, x_seq, sentiment_scalar):
        lstm_out = self.lstm(x_seq)
        
        if sentiment_scalar.dim() == 1:
            sentiment_scalar = sentiment_scalar.unsqueeze(1)
            
        combined_input = torch.cat((lstm_out, sentiment_scalar), dim=1)
        final_output = self.fuzzy(combined_input)
        
        return final_output