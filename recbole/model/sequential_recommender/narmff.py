# -*- coding: utf-8 -*-

# UPDATE WITH SIDE INFORMATION
# @Time   : 2020/4/15
# @Author : Martin Dimitrov
# @Email  : mdimitrov123@gmail.com

# @Time   : 2020/8/25 19:56
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/9/15, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com




r"""
NARM
################################################

Reference:
    Jing Li et al. "Neural Attentive Session-based Recommendation." in CIKM 2017.

Reference code:
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class NARMFF(SequentialRecommender):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self, config, dataset):
        super(NARMFF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.dropout_probs = config['dropout_probs']
        self.device = config['device']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        self.loss_type = config['loss_type']
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(self.embedding_size*3, self.hidden_size, self.n_layers, bias=False, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2*self.hidden_size, self.embedding_size, bias=False)
        self.concat_layer = nn.Linear(self.hidden_size * (1 + self.num_feature_field), self.hidden_size)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        #torch.Size([2048, 50, 64])

        #adapted from SASRecF
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        feature_table = torch.cat(feature_table, dim=-2)
        table_shape = feature_table.shape
        #([2048, 50, 1, 128])
        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))
        #torch.Size([2048, 50, 128])
        output_concat = torch.cat((item_seq_emb, feature_emb), -1)  # [B Len 2*H
        output = self.emb_dropout(output_concat)
        gru_out, _ = self.gru(output)
        #torch.Size([2048, 50, 128])
        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        #torch.Size([2048, 128])
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        #torch.Size([2048, 50, 128])
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores