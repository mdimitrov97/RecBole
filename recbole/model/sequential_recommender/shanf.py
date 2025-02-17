# -*- coding: utf-8 -*-

# UPDATE WITH SIDE INFORMATION
# @Time   : 2020/4/15
# @Author : Martin Dimitrov
# @Email  : mdimitrov123@gmail.com

# @Time     : 2020/11/20 22:33
# @Author   : Shao Weiqi
# @Reviewer : Lin Kun
# @Email    : shaoweiqi@ruc.edu.cn

r"""
SHAN
################################################

Reference:
    Ying, H et al. "Sequential Recommender System based on Hierarchical Attention Network."in IJCAI 2018


"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class SHANF(SequentialRecommender):
    r"""
    SHAN exploit the Hierarchical Attention Network to get the long-short term preference
    first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose

    """

    def __init__(self, config, dataset):

        super(SHANF, self).__init__(config, dataset)

        # load the dataset information
        self.n_users = dataset.num(self.USER_ID)
        self.device = config['device']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        # load the parameter information
        self.embedding_size = config["embedding_size"]
        self.short_item_length = config["short_item_length"]  # the length of the short session items
        assert self.short_item_length <= self.max_seq_length, "short_item_length can't longer than the max_seq_length"
        self.reg_weight = config["reg_weight"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)

        self.long_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.long_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.embedding_size),
                a=-np.sqrt(3 / self.embedding_size),
                b=np.sqrt(3 / self.embedding_size)
            ),
            requires_grad=True
        ).to(self.device)
        self.long_short_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.long_short_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.embedding_size),
                a=-np.sqrt(3 / self.embedding_size),
                b=np.sqrt(3 / self.embedding_size)
            ),
            requires_grad=True
        ).to(self.device)

        self.relu = nn.ReLU()
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.embedding_size, self.selected_features, self.pooling_mode, self.device
        )
        self.concat_layer = nn.Linear(self.embedding_size * (1 + self.num_feature_field), self.embedding_size)
        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init the parameter of the model
        self.apply(self.init_weights)

    def reg_loss(self, user_embedding, item_embedding):

        reg_1, reg_2 = self.reg_weight
        loss_1 = reg_1 * torch.norm(self.long_w.weight, p=2) + reg_1 * torch.norm(self.long_short_w.weight, p=2)
        loss_2 = reg_2 * torch.norm(user_embedding, p=2) + reg_2 * torch.norm(item_embedding, p=2)

        return loss_1 + loss_2

    def inverse_seq_item(self, seq_item, seq_item_len):
        """
        inverse the seq_item, like this
            [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
        """
        seq_item = seq_item.cpu().numpy()
        seq_item_len = seq_item_len.cpu().numpy()
        new_seq_item = []
        for items, length in zip(seq_item, seq_item_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_seq_item.append(seqs)
        seq_item = torch.tensor(new_seq_item, dtype=torch.long, device=self.device)

        return seq_item

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0., 0.01)
        elif isinstance(module, nn.Linear):
            uniform_(module.weight.data, -np.sqrt(3 / self.embedding_size), np.sqrt(3 / self.embedding_size))
        elif isinstance(module, nn.Parameter):
            uniform_(module.data, -np.sqrt(3 / self.embedding_size), np.sqrt(3 / self.embedding_size))
            print(module.data)

    def forward(self, seq_item, user, seq_item_len):

        seq_item = self.inverse_seq_item(seq_item, seq_item_len)

        seq_item_embedding = self.item_embedding(seq_item)
        user_embedding = self.user_embedding(user)

        #adapted from SASRec
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, seq_item)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        feature_table = torch.cat(feature_table, dim=-2)
        table_shape = feature_table.shape
        # print("Table shape",feature_table.shape)
        # torch.Size([2048, 50, 1, 64])
        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))
        # print(feature_emb.shape) #2048x50x64
        input_concat = torch.cat((seq_item_embedding, feature_emb), -1)  # [B 1+field_num*H]
        # print(input_concat.shape) #2048x50x128

        input_emb = self.concat_layer(input_concat)


        # get the mask
        mask = seq_item.data.eq(0)
        long_term_attention_based_pooling_layer = self.long_term_attention_based_pooling_layer(
            input_emb, user_embedding, mask
        )
        # batch_size * 1 * embedding_size

        short_item_embedding = seq_item_embedding[:, -self.short_item_length:, :]
        mask_long_short = mask[:, -self.short_item_length:]
        batch_size = mask_long_short.size(0)
        x = torch.zeros(size=(batch_size, 1)).eq(1).to(self.device)
        mask_long_short = torch.cat([x, mask_long_short], dim=1)
        # batch_size * short_item_length * embedding_size
        long_short_item_embedding = torch.cat([long_term_attention_based_pooling_layer, short_item_embedding], dim=1)
        # batch_size * 1_plus_short_item_length * embedding_size

        long_short_item_embedding = self.long_and_short_term_attention_based_pooling_layer(
            long_short_item_embedding, user_embedding, mask_long_short
        )
        # batch_size * embedding_size

        return long_short_item_embedding

    def calculate_loss(self, interaction):

        seq_item = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        user_embedding = self.user_embedding(user)
        seq_output = self.forward(seq_item, user, seq_item_len)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items)
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss + self.reg_loss(user_embedding, pos_items_emb)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss + self.reg_loss(user_embedding, pos_items_emb)

    def predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user, seq_item_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        seq_item_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, user, seq_item_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores

    def long_and_short_term_attention_based_pooling_layer(self, long_short_item_embedding, user_embedding, mask=None):
        """

        fusing the long term purpose with the short-term preference
        """
        long_short_item_embedding_value = long_short_item_embedding

        long_short_item_embedding = self.relu(self.long_short_w(long_short_item_embedding) + self.long_short_b)
        long_short_item_embedding = torch.matmul(long_short_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            long_short_item_embedding.masked_fill_(mask, -1e9)
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding)
        long_short_item_embedding = torch.mul(long_short_item_embedding_value,
                                              long_short_item_embedding.unsqueeze(2)).sum(dim=1)

        return long_short_item_embedding

    def long_term_attention_based_pooling_layer(self, seq_item_embedding, user_embedding, mask=None):
        """

        get the long term purpose of user
        """
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.relu(self.long_w(seq_item_embedding) + self.long_b)
        user_item_embedding = torch.matmul(seq_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            user_item_embedding.masked_fill_(mask, -1e9)
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding)
        user_item_embedding = torch.mul(seq_item_embedding_value,
                                        user_item_embedding.unsqueeze(2)).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding
