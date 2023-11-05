import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import layer_utils as Layer


class Baseline(nn.Module):
    def __init__(self, config, model_parameters):
        super(Baseline, self).__init__()
        self.config = config

        if config.data_type == 'vie':
            print('phobert')
            self.encoder = Layer.PhoBERTCell(config.path.bert_model_path)
        else:
            print('bert')
            self.encoder = Layer.BERTCell(config.path.bert_model_path)

        self.embedding_dropout = nn.Dropout(model_parameters['embed_dropout'])
        self.weight_dropout = nn.Dropout(model_parameters['embed_dropout'])
        config.hidden_size = self.encoder.hidden_size

        # define hyper-parameters.
        if "loss_factor" in model_parameters:
            self.gamma = model_parameters['loss_factor']
        else:
            self.gamma = 1

        # define sentence classification
        self.sent_linear = nn.Linear(self.encoder.hidden_size, 2)

        # define mapping full-connect layer.
        self.W = nn.ModuleList()
        for i in range(4):
            self.W.append(copy.deepcopy(nn.Linear(self.encoder.hidden_size, len(config.val.norm_id_map))))

        # define multi-crf decode the sequence.
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(copy.deepcopy(Layer.CRFCell(len(config.val.norm_id_map), batch_first=True)))

    def forward(self, input_ids, attn_mask, comparative_label=None, elem_label=None, result_label=None):
        # get token embedding.
        token_embedding, pooled_output = self.encoder(input_ids, attn_mask)

        batch_size, sequence_length, _ = token_embedding.size()

        final_embedding = self.embedding_dropout(token_embedding)
        class_embedding = self.embedding_dropout(pooled_output)

        # linear mapping.
        multi_sequence_prob = [self.weight_dropout(self.W[index](final_embedding)) for index in range(len(self.W))]
        sent_class_prob = self.sent_linear(class_embedding)

        # decode sequence label.
        elem_output = []
        for index in range(3):
            if elem_label is None:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, None))
            else:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, elem_label[:, index, :]))

        # result extract sequence label.
        result_output = self.decoder[3](multi_sequence_prob[3], attn_mask, result_label)

        if elem_label is None and result_label is None:
            _, sent_output = torch.max(torch.softmax(sent_class_prob, dim=1), dim=1)

            elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)

            elem_feature = multi_sequence_prob
            elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
            elem_feature = torch.cat(elem_feature, dim=0).permute(1, 0, 2, 3)

            # elem_feature: [B, 3, N, feature_dim]
            # result_feature: [B, N, feature_dim]
            return token_embedding, elem_feature, elem_output, result_output, sent_output

        # calculate sent loss and crf loss.
        sent_loss = F.cross_entropy(sent_class_prob, comparative_label.view(-1))
        crf_loss = sum(elem_output) + result_output

        # according different model type to get different loss type.
        if self.config.model_type == "classification":
            return sent_loss

        elif self.config.model_type == "extraction":
            return crf_loss

        else:
            return sent_loss + self.gamma * crf_loss


class LSTMModel(nn.Module):
    def __init__(self, config, model_parameters, vocab, weight=None):
        super(LSTMModel, self).__init__()
        self.config = config

        # input embedding.
        if weight is not None:
            self.input_embed = nn.Embedding(len(vocab) + 10, config.input_size).from_pretrained(weight)
        else:
            self.input_embed = nn.Embedding(len(vocab) + 10, config.input_size)

        self.encoder = Layer.LSTMCell(
            config.input_size, config.hidden_size, config.num_layers,
            config.device, batch_first=True, bidirectional=True
        )
        self.embedding_dropout = nn.Dropout(model_parameters['embed_dropout'])

        # define hyper-parameters.
        if "loss_factor" in model_parameters:
            self.gamma = model_parameters['loss_factor']
        else:
            self.gamma = 0

        # define sentence classification
        self.sent_linear = nn.Linear(self.encoder.hidden_size * 2, 2)

        # define mapping full-connect layer.
        self.W = nn.ModuleList()
        for i in range(4):
            self.W.append(copy.deepcopy(nn.Linear(self.encoder.hidden_size * 2, len(config.val.norm_id_map))))

        # define multi-crf decode the sequence.
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(copy.deepcopy(Layer.CRFCell(len(config.val.norm_id_map), batch_first=True)))

    def forward(self, input_ids, attn_mask, comparative_label=None, elem_label=None, result_label=None):
        # get token embedding.

        input_embedding = self.input_embed(input_ids)
        token_embedding, pooled_output = self.encoder(input_embedding)

        batch_size, sequence_length, _ = token_embedding.size()

        final_embedding = self.embedding_dropout(token_embedding)
        class_embedding = self.embedding_dropout(pooled_output)

        # linear mapping.
        multi_sequence_prob = [self.W[index](final_embedding) for index in range(len(self.W))]
        sent_class_prob = self.sent_linear(class_embedding)

        # decode sequence label.
        elem_output = []
        for index in range(3):
            if elem_label is None:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, None))
            else:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, elem_label[:, index, :]))

        # result extract sequence label.
        result_output = self.decoder[3](multi_sequence_prob[3], attn_mask, result_label)

        if elem_label is None and result_label is None:
            _, sent_output = torch.max(torch.softmax(sent_class_prob, dim=1), dim=1)

            elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)

            elem_feature = multi_sequence_prob
            elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
            elem_feature = torch.cat(elem_feature, dim=0).permute(1, 0, 2, 3)

            # elem_feature: [B, 3, N, feature_dim]
            # result_feature: [B, N, feature_dim]
            return token_embedding, elem_feature, elem_output, result_output, sent_output

        # calculate sent loss and crf loss.
        sent_loss = F.cross_entropy(sent_class_prob, comparative_label.view(-1))
        crf_loss = sum(elem_output) + result_output

        print(sent_loss, crf_loss)
        # according different model type to get different loss type.
        if self.config.model_type == "classification":
            return sent_loss

        elif self.config.model_type == "extraction":
            return crf_loss

        else:
            return sent_loss + self.gamma * crf_loss


class LogisticClassifier(nn.Module):
    def __init__(self, config, feature_dim, class_num=2, dropout=0.1, weight=None):
        super(LogisticClassifier, self).__init__()
        self.config = config
        self.class_num = class_num

        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, class_num)
        self.weight = weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_representation, pair_label=None, return_proba=False):
        predict_label = self.fc(pair_representation.view(-1, self.feature_dim))
        predict_label = self.dropout(predict_label)

        # weight = torch.tensor([1, 1, 1, 1]).float().to(self.config.device)
        # calculate loss.
        if pair_label is not None:
            if self.weight is not None:
                self.weight = self.weight.to(self.config.device)
                criterion = nn.CrossEntropyLoss(weight=self.weight)
            else:
                criterion = nn.CrossEntropyLoss()
            
            return criterion(predict_label, pair_label.view(-1))
        else:
            if return_proba:
                return F.softmax(predict_label, dim=-1)
            return torch.max(F.softmax(predict_label, dim=-1), dim=-1)[-1]
        

class MLPClasifier(nn.Module):
    hidden_dim = 256
    def __init__(self, config, feature_dim, class_num=2, dropout=0.1, weight=None):
        super(MLPClasifier, self).__init__()
        self.config = config
        self.class_num = class_num

        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            # nn.LayerNorm(self.hidden_dim),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.class_num)
        )
        self.weight = weight

    def forward(self, pair_representation, pair_label=None, return_proba=False):
        predict_label = self.mlp(pair_representation.view(-1, self.feature_dim))

        # weight = torch.tensor([1, 1, 1, 1]).float().to(self.config.device)
        # calculate loss.
        if pair_label is not None:
            if self.weight is not None:
                self.weight = self.weight.to(self.config.device)
                criterion = nn.CrossEntropyLoss(weight=self.weight)
            else:
                criterion = nn.CrossEntropyLoss()
            
            return criterion(predict_label, pair_label.view(-1))
        else:
            if return_proba:
                return F.softmax(predict_label, dim=-1)
            return torch.max(F.softmax(predict_label, dim=-1), dim=-1)[-1]
        
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(den * pos)
        pos_embedding[:, 1::2] = torch.cos(den * pos)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :]) 
    
class transformer_classifier(nn.Module):
    def __init__(self, config, feature_dim, num_head, num_layer, num_class=2, dropout=0.1, weight=None):
        super(transformer_classifier, self).__init__()

        self.config = config
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.num_head = num_head
        self.weight = weight
        self.tmp_feature_dim = (self.feature_dim // self.num_head) * self.num_head
        self.linear = nn.Linear(self.feature_dim, self.tmp_feature_dim)
        # self.pos_encoding = PositionalEncoding(self.feature_dim, dropout)
        self.token_type_embedding = nn.Embedding(5, self.tmp_feature_dim)
        

        self.attentions = nn.ModuleList([
            copy.deepcopy(nn.MultiheadAttention(embed_dim=self.tmp_feature_dim, num_heads=self.num_head, batch_first=True)) for i in range(num_layer)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.tmp_feature_dim, self.num_class)
        # self.attention_2 = nn.MultiheadAttention(self.feature_dim, num_head=self.num_head)

    def forward(self, comp_feature, token_type_feature, padding_mask=None, label=None):
        # if torch.equal(token_type_feature, torch.zeros_like(token_type_feature))
        # pos_encode_bert_feature = self.pos_encoding(comp_feature)
        norm_comp_feature = self.linear(comp_feature)
        norm_token_type_feature = self.token_type_embedding(token_type_feature)
        
        # fushion_feature = pos_encode_bert_feature + norm_token_type_feature
        # print(comp_feature.shape, norm_token_type_feature.shape)
        fushion_feature = norm_comp_feature + norm_token_type_feature
        if label is None:
            print(f"transformers_input: {fushion_feature.shape}")
        """
        fushion_feature: batch_size, seq_length, embed_dim
        """
        for attention_layer in self.attentions:
            fushion_feature = attention_layer(fushion_feature, fushion_feature, fushion_feature, key_padding_mask=padding_mask)[0]
        # attn_output = self.attentions(fushion_feature, fushion_feature, fushion_feature)

        pooled_output = fushion_feature.permute(1, 0, 2)[0]
        predicted_label = self.classifier(pooled_output)
        """
        pooled_output: batch_size, embed_dim
        predicted_label: batch_size, num_class
        """

        if label is not None:
            if self.weight is not None:
                self.weight = self.weight.to(self.config.device)
                criterion = nn.CrossEntropyLoss(weight=self.weight)
            else:
                criterion = nn.CrossEntropyLoss()

            return criterion(predicted_label, label.view(-1))
        else:
            print(f"predicted_label: {predicted_label.shape}")
            return torch.max(F.softmax(predicted_label, dim=-1), dim=-1)[-1]

