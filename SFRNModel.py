import torch
import torch.nn as nn
from transformers import AutoModel
from Constants import *


class SFRNModel(nn.Module):
    def __init__(self):
        super(SFRNModel, self).__init__()
        # Define the pre-trained model and tokenizer
        #self.device = DEVICE
        self.bert = AutoModel.from_pretrained(hyperparameters['model_name'])
        self.dropout = torch.nn.Dropout(hyperparameters['hidden_dropout_prob'])
#         self.bert = transformers.BertModel.from_pretrained(model_name, config=config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Define the number of labels in the classification task
        num_labels = hyperparameters['num_labels']
        # A single layer classifier added on top of BERT to fine tune for binary classification
        mlp_hidden = hyperparameters['mlp_hidden']
        self.g = nn.Sequential(
            nn.Linear(hyperparameters['hidden_dim'], mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            # nn.Linear(mlp_hidden, mlp_hidden),
            # nn.ReLU(),
            # nn.Linear(mlp_hidden, mlp_hidden),
            # nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            # nn.Linear(mlp_hidden, mlp_hidden),
            # nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, num_labels),
        )

        self.alpha = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.beta = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Forward pass through pre-trained BERT
        #print('input_ids: ', input_ids.size())
        #print('attention_mask: ', attention_mask.size())
        outputs = self.bert(input_ids.squeeze(), attention_mask=attention_mask.squeeze())
        pooled_output = outputs[-1]
        #print('pooled_output: ', pooled_output.size())
        pooled_output = self.dropout(pooled_output)
        #print("pooled_output: {}".format(pooled_output.size()))
        g_t = self.g(pooled_output)
        g_t = self.alpha(g_t) * g_t + self.beta(g_t)
        #print("g_t: {}".format(g_t.size()))
        g = g_t.sum(0)
        #print("g: {}".format(g.size()))
        #g = g_t.sum(1) + g_t.prod(1)
        output = self.f(g.unsqueeze(0))
        #print("f: {}".format(output.size()))
        logits = torch.softmax(output, dim=1)
        return logits
        # return self.classifier(pooled_output)[0].unsqueeze(0)
