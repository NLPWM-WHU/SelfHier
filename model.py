from typing import Dict, List, Optional, Tuple
import copy
import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import random
import json
from numpy.core.numeric import Inf
import math
import torch.nn as nn

class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model, max_seq_len=20):
        super(PositionalEncoding, self).__init__()
        self.device = device
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=True)

    def forward(self, batch_len, start, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(start + 1, start + seq_len + 1)) for _ in range(batch_len)]).to(self.device)
        return self.position_encoding(input_pos).transpose(0, 1)

class PositionalEncoding1(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, args, dictionary, mode=None, true_triples=None):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
        except:
            raise ImportError('Transformer module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.mode = mode
        self.ninp = args.embedding_dim
        self.args = args
        self.device = "cuda:{0}".format(args.cuda) if torch.cuda.is_available() else "cpu"
        self.pos_encoder = PositionalEncoding(self.device, self.ninp)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.embedding_dim, nhead=4, dim_feedforward=args.hidden_size, dropout=args.dropout)
        self.enencoder = nn.TransformerEncoder(encoder_layers, args.num_layers)
        self.ntoken = len(dictionary)
        self.padding_idx = dictionary.pad()
        if args.use_pretrained_emb:
            if mode == 'cluster':
                cluster2emb = []
                with open(args.dataset + '/cluster2vec' + args.K + '.bern', "r") as f:
                    for line in f:
                        cluster2emb.append([float(value) for value in line.split()])
                clusteremb = torch.tensor(cluster2emb)
                otheremb = torch.randn(len(dictionary)-len(clusteremb),self.ninp)
                if args.xavier_all:
                    emb = torch.cat([otheremb, clusteremb])
                    torch.nn.init.xavier_normal_(emb)
                else:
                    torch.nn.init.xavier_normal_(otheremb)
                    emb = torch.cat([otheremb,clusteremb])
            else:
                entity2emb = []
                with open(args.dataset + '/entity2vec.bern', "r") as f:
                    for line in f:
                        entity2emb.append([float(value) for value in line.split()])
                entityemb = torch.tensor(entity2emb)
                otheremb = torch.randn(len(dictionary)-len(entityemb),self.ninp)
                if args.xavier_all:
                    emb = torch.cat([otheremb, entityemb])
                    torch.nn.init.xavier_normal_(emb)
                else:
                    torch.nn.init.xavier_normal_(otheremb)
                    emb = torch.cat([otheremb,entityemb])
            self.encoder = nn.Embedding(self.ntoken, self.ninp).from_pretrained(emb)
            self.encoder.weight.requires_grad = True
        else:
            self.encoder = nn.Embedding(self.ntoken, self.ninp)
        self.fc = torch.nn.Linear(self.ninp, self.ninp)
        self.dictionary = dictionary
        self.glue = GELU()
        self.label_smooth = args.label_smooth

        #self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        xavier_normal_(self.encoder.weight.data)

    def logits(self, source, prev_outputs, **unused):
        bsz, src_len = source.shape
        out_len = prev_outputs.size(1)
        device = source.device
        source = source.transpose(0, 1)
        source = self.encoder(source)
        source += self.pos_encoder(bsz, 0, src_len)
        mask = self._generate_square_subsequent_mask(prev_outputs.size(-1))
        prev_outputs = prev_outputs.transpose(0, 1)
        prev_outputs = self.encoder(prev_outputs)
        prev_outputs += self.pos_encoder(bsz, src_len, out_len)
        rel_emb = source[-1]
        prev_outputs += rel_emb
        if self.args.encoder:
            enmask = torch.zeros(out_len + src_len, out_len + src_len)
            enmask[:, src_len:] = float("-inf")
            enmask[src_len:, src_len:] = mask
            enmask = enmask.to(device)
            output = self.enencoder(torch.cat((source, prev_outputs), dim=0), mask=enmask)[src_len:, :, :].transpose(0, 1)
        else:
            mask = mask.to(device)
            output = self.endecoder(source, prev_outputs, tgt_mask=mask).transpose(0, 1)
        logits = torch.mm(self.glue(self.fc(output)).view(-1, self.ninp), self.encoder.weight.transpose(0, 1)).view(bsz, out_len, -1)
        return logits

    def get_loss(self, source, prev_outputs, target, mask):
        logits = self.logits(source, prev_outputs)
        lprobs = F.log_softmax(logits, dim=-1).to(self.device)
        self.entity_prob = logits
        loss = -(self.label_smooth * torch.gather(input=lprobs, dim=-1, index=target.unsqueeze(-1)).squeeze() \
            + (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1)) * mask
        loss = loss.sum() / mask.sum()
        return loss, logits

    def get_entity_matrix(self, entity_vocab_len, cluster_vocab_len):
        path =  self.args.dataset + '/entity2clusterid_' + self.args.K + '.txt'
        dict = {}
        with open(path, 'r') as f:
            for line in f:
                entity, cluster = line.strip().split()
                dict[int(entity)] = int(cluster)
        length = len(dict)
        matrix = torch.zeros(entity_vocab_len, cluster_vocab_len)
        for i in range(entity_vocab_len - length):
            matrix[i][i] = 1
        for i in range(entity_vocab_len - length, entity_vocab_len):
            matrix[i][entity_vocab_len - length + dict[i - (cluster_vocab_len-int(self.args.K))]] = 1
        matrix = matrix.transpose(0,1)
        matrix = matrix.to_sparse()
        self.matrix = matrix.to(self.device)

    def get_relation_matrix(self, entity_vocab_len, relation_vocab_len, entity_num):
        path =  self.args.dataset + '/multi_relation2id.txt'
        dict = {}
        with open(path, 'r') as f:
            for line in f:
                realtion, relation_ids = line.strip().split('\t')
                dict[int(relation_ids)] = int(int(relation_ids)/3)
        relation_num = int((relation_vocab_len - entity_vocab_len)/3)
        matrix = torch.zeros(relation_vocab_len, entity_vocab_len)
        for i in range(6+relation_num):
            matrix[i][i] = 1
        for i in range(6+relation_num, relation_vocab_len - entity_num):
            matrix[i][6 + dict[i-6-relation_num]] = 1
        for i in range(entity_num):
            matrix[-i][-i] = 1
        matrix = matrix.transpose(0,1)
        matrix = matrix.to_sparse()
        self.matrix = matrix.to(self.device)

    def get_cross_loss1(self, cluster_prob):
        criterion = nn.CosineEmbeddingLoss(reduction='mean')
        e_prob = self.entity_prob.view(-1,self.entity_prob.size()[-1]).transpose(0,1)
        aaa1 = torch.spmm(self.matrix, e_prob).transpose(0,1)\
            .view(self.entity_prob.size()[0],self.entity_prob.size()[1],-1)
        bbb1 = cluster_prob.to(self.device)
        aaa2 = aaa1.view(-1,aaa1.size()[-1])
        bbb2 = bbb1.view(-1,bbb1.size()[-1])
        target = torch.ones(aaa2.size()[0]).to(self.device)
        output = criterion(aaa2,bbb2,target)
        return output, self.matrix

    def get_cross_loss2(self, cluster_prob):
        e_prob = self.entity_prob.view(-1,self.entity_prob.size()[-1]).transpose(0,1)
        entity_to_cluster_prob = torch.spmm(self.matrix, e_prob)
        entity_to_cluster_prob = entity_to_cluster_prob.transpose(0,1).view(self.entity_prob.size()[0],self.entity_prob.size()[1],-1)
        if self.mode == 'relation':
            entity_tensor1 = entity_to_cluster_prob[:, 0:cluster_prob.size()[-2]-1:2, :]
            entity_tenosr2 = cluster_prob[:, 0:cluster_prob.size()[-2]-1:2, :]
        else:
            entity_tensor1 = entity_to_cluster_prob[:,1:cluster_prob.size()[-2]:2,:]
            entity_tenosr2 = cluster_prob[:,1:cluster_prob.size()[-2]:2,:]
        entity_tenosr3 = entity_tensor1.contiguous().view(-1, entity_tensor1.size()[-1])
        entity_tenosr4 = entity_tenosr2.contiguous().view(-1, entity_tenosr2.size()[-1])
        criterion = nn.CosineEmbeddingLoss(reduction='mean')
        target = torch.ones(entity_tenosr3.size()[0]).to(self.device)
        output = criterion(entity_tenosr3, entity_tenosr4, target)
        return  output, self.matrix

