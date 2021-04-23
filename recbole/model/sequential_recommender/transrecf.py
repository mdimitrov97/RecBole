# -*- coding: utf-8 -*-

# UPDATE WITH SIDE INFORMATION
# @Time   : 2020/4/15
# @Author : Martin Dimitrov
# @Email  : mdimitrov123@gmail.com

# @Time    : 2020/9/14 17:01
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
TransRec
################################################

Reference:
    Ruining He et al. "Translation-based Recommendation." In RecSys 2017.

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss, RegLoss
from recbole.utils import InputType
from recbole.model.layers import FeatureSeqEmbLayer

class TransRecF(SequentialRecommender):
    r"""
    TransRec is translation-based model for sequential recommendation.
    It assumes that the `prev. item` + `user`  = `next item`.
    We use the Euclidean Distance to calculate the similarity in this implementation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(TransRecF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']

        # load dataset info
        self.n_users = dataset.user_num

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.bias = nn.Embedding(self.n_items, 1, padding_idx=0)  # Beta popularity bias
        self.T = nn.Parameter(torch.zeros(self.embedding_size))  # average user representation 'global'
        self.selected_features = config['selected_features']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.pooling_mode = config['pooling_mode']
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layer_norm_eps = config['layer_norm_eps']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_loss = RegLoss()
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.embedding_size, self.selected_features, self.pooling_mode, self.device
        )
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.concat_layer = nn.Linear(self.embedding_size * (1 + self.num_feature_field), self.embedding_size)
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _l2_distance(self, x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1, keepdim=True))  # [B 1]

    def gather_last_items(self, item_seq, gather_index):
        """Gathers the last_item at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1)
        last_items = item_seq.gather(index=gather_index, dim=1)  # [B 1]
        return last_items.squeeze(-1)  # [B]

    def forward(self, user, item_seq, item_seq_len):
        # the last item at the last position
        last_items = self.gather_last_items(item_seq, item_seq_len - 1)  # [B]
        user_emb = self.user_embedding(user)  # [B H]
        last_items_emb = self.item_embedding(last_items)  # [B H]
        #print(last_items_emb.shape) #2048x64
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        #adapted from SASRec
        feature_table = torch.cat(feature_table,dim=-2)
        table_shape = feature_table.shape
        #print(table_shape) #2048x50x1x64
        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))
        feature_emb = feature_emb.narrow(1,0,1).view(table_shape[0],embedding_size)
        #print(feature_emb.shape) #2048x50x64
        input_concat = torch.cat((last_items_emb, feature_emb), -1)  # [B 1+field_num*H]
        input_concat = self.concat_layer(input_concat)
        #input_emb = self.LayerNorm(input_concat)
        #input_concat = self.dropout(input_emb)
        #print(input_concat.shape) #2048x50x128
        #T = self.T.expand_as(user_emb)  # [B H]
        seq_output = user_emb + input_concat  # [B H]
        return seq_output

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        pos_items = interaction[self.POS_ITEM_ID]  # [B]
        neg_items = interaction[self.NEG_ITEM_ID]  # [B] sample 1 negative item

        pos_items_emb = self.item_embedding(pos_items)  # [B H]
        neg_items_emb = self.item_embedding(neg_items)

        pos_bias = self.bias(pos_items)  # [B 1]
        neg_bias = self.bias(neg_items)

        pos_score = pos_bias - self._l2_distance(seq_output, pos_items_emb)
        neg_score = neg_bias - self._l2_distance(seq_output, neg_items_emb)

        bpr_loss = self.bpr_loss(pos_score, neg_score)
        item_emb_loss = self.emb_loss(self.item_embedding(pos_items).detach())
        user_emb_loss = self.emb_loss(self.user_embedding(user).detach())
        bias_emb_loss = self.emb_loss(self.bias(pos_items).detach())

        reg_loss = self.reg_loss(self.T)
        return bpr_loss + item_emb_loss + user_emb_loss + bias_emb_loss + reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)  # [B H]
        test_bias = self.bias(test_item)  # [B 1]

        scores = test_bias - self._l2_distance(seq_output, test_item_emb)  # [B 1]
        scores = scores.squeeze(-1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]  # [B]
        item_seq = interaction[self.ITEM_SEQ]  # [B Len]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(user, item_seq, item_seq_len)  # [B H]

        test_items_emb = self.item_embedding.weight  # [item_num H]
        test_items_emb = test_items_emb.repeat(seq_output.size(0), 1, 1)  # [user_num item_num H]

        user_hidden = seq_output.unsqueeze(1).expand_as(test_items_emb)  # [user_num item_num H]
        test_bias = self.bias.weight  # [item_num 1]
        test_bias = test_bias.repeat(user_hidden.size(0), 1, 1)  # [user_num item_num 1]

        scores = test_bias - self._l2_distance(user_hidden, test_items_emb)  # [user_num item_num 1]
        scores = scores.squeeze(-1)  # [B n_items]
        return scores
