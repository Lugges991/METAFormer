
import torch
import torch.nn as nn
import math


class SAT(nn.Module):
    def __init__(self, feat_dim, d_model=128, dim_feedforward=128, num_encoder_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.encoder = EncoderBlock(
            feat_dim, d_model, dim_feedforward, num_encoder_layers, num_heads, dropout)
        self.act = nn.GELU()
        self.do = nn.Dropout(p=dropout)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.act(x)
        x = self.do(x)
        x = self.head(x)

        return x


class METAFormer(nn.Module):
    def __init__(self, d_model=128, dim_feedforward=128, num_encoder_layers=2, num_heads=8, dropout=0.1, state_dict=None):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.aal_encoder = EncoderBlock(
            6670, d_model, dim_feedforward, num_encoder_layers, num_heads, dropout)
        self.cc200_encoder = EncoderBlock(
            19900, d_model, dim_feedforward, num_encoder_layers, num_heads, dropout)
        self.dos160_encoder = EncoderBlock(
            12880, d_model, dim_feedforward, num_encoder_layers, num_heads, dropout)
        self.aal_act = nn.GELU()
        self.cc200_act = nn.GELU()
        self.dos160_act = nn.GELU()

        self.aal_do = nn.Dropout(p=dropout)
        self.cc200_do = nn.Dropout(p=dropout)
        self.dos160_do = nn.Dropout(p=dropout)

        self.aal_head = nn.Linear(d_model, 2)
        self.cc200_head = nn.Linear(d_model, 2)
        self.dos160_head = nn.Linear(d_model, 2)

    def forward(self, aal, cc200, dos160):
        aal = self.aal_encoder(aal)
        cc200 = self.cc200_encoder(cc200)
        dos160 = self.dos160_encoder(dos160)

        aal = self.aal_act(aal)
        cc200 = self.cc200_act(cc200)
        dos160 = self.dos160_act(dos160)

        aal = self.aal_do(aal)
        cc200 = self.cc200_do(cc200)
        dos160 = self.dos160_do(dos160)

        aal = self.aal_head(aal)
        cc200 = self.cc200_head(cc200)
        dos160 = self.dos160_head(dos160)

        output = (aal + cc200 + dos160) / 3

        return output


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.inp_emb = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, x):
        x = self.inp_emb(x) * math.sqrt(self.d_model)
        x = self.encoder(x)
        return x
