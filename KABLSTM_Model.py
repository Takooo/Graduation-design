#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/1
# @Author  : Takoo
# @File    : KABLSTM_Model.py
# @Software: PyCharm

from typing import Dict
import torch
import numpy as np
import math
from allennlp.data.vocabulary import Vocabulary
from configparser import ConfigParser
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

MAX_LEN = 250

class BasicClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 ent_embeddings,
                 vocab: Vocabulary,
                 config: ConfigParser) -> None:
        super().__init__(vocab)
        self.encoder = ElmoLstm(300, 64, 4, 1)
        self.embeddings_feature = int(config.get('Model', 'embeddings_feature'))
        self.sentence_max_length = int(config.get('Model', 'sentence_max_length'))
        self.attention_size = int(config.get('Model', 'attention_size'))
        self.out_hidenlayer_feature = int(config.get('Model', 'out_hidenlayer_feature'))
        self.conv1_size = int(config.get('Model', 'conv1_size'))
        self.conv2_size = int(config.get('Model', 'conv2_size'))
        self.word_embeddings = word_embeddings
        self.ent_embeddings = ent_embeddings
        self.hidden_layer = torch.nn.Linear(128,
                                          out_features=self.embeddings_feature)
        self.cnn1 = torch.nn.Conv1d(self.sentence_max_length, self.sentence_max_length, 2)
        self.cnn1.bias.data.fill_(0.1)
        torch.nn.init.xavier_normal_(self.cnn1.weight)
        self.cnn2 = torch.nn.Conv1d(self.sentence_max_length, self.sentence_max_length, 3)
        self.cnn2.bias.data.fill_(0.1)
        torch.nn.init.xavier_normal_(self.cnn2.weight)
        self.softmax = torch.nn.Softmax(-1)
        sim_size = self.embeddings_feature*3-self.conv1_size-self.conv2_size+2
        self.sim_layer = torch.nn.Linear(sim_size, sim_size)
        out_layer_input = sim_size*2+1
        self.out_layer = torch.nn.Linear(out_layer_input, self.out_hidenlayer_feature)
        self._classification_layer = torch.nn.Linear(self.out_hidenlayer_feature, 2)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(0)
        self._loss = torch.nn.CrossEntropyLoss()

        self.W = {
            'Whm': torch.autograd.Variable(torch.FloatTensor(self.embeddings_feature, self.attention_size).uniform_(-1, 1), requires_grad=True),
            'Wem': torch.autograd.Variable(torch.FloatTensor(self.embeddings_feature, self.attention_size).uniform_(-1, 1), requires_grad=True),
            'Wms': torch.autograd.Variable(torch.FloatTensor(self.attention_size, 1).uniform_(-1, 1), requires_grad=True)
        }

        self.U1 = torch.autograd.Variable(torch.nn.init.xavier_uniform_(torch.Tensor(self.embeddings_feature, self.embeddings_feature)), requires_grad=True)
        self.U2 = torch.autograd.Variable(torch.nn.init.xavier_uniform_(torch.Tensor(125, 125)), requires_grad=True)

        if torch.cuda.is_available():
            self.W['Whm'] = self.W['Whm'].cuda()
            self.W['Wem'] = self.W['Wem'].cuda()
            self.W['Wms'] = self.W['Wms'].cuda()
            self.U1 = self.U1.cuda()
            self.U2 = self.U2.cuda()

    def get_label_embedding(self, sentence_label):
        label_embed = np.zeros((len(sentence_label), len(sentence_label[0]), len(sentence_label[0][0]), self.embeddings_feature))
        for i, sentence in enumerate(sentence_label):
            for j, piece in enumerate(sentence):
                for k, num in enumerate(piece):
                    label_embed[i][j][k] = self.ent_embeddings[num]
        return torch.Tensor(label_embed)

    def matrix_diag(self, diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result

    def kb_module(self, H: torch.Tensor, E: torch.Tensor, W):
        # n, p, k, e = H.size(0), H.size(1), H.size(2), H.size(3)
        Hinit = H.clone()
        Ent = E.clone()
        Hinit = Hinit.reshape(H.size(0)*H.size(1)*H.size(2), H.size(3))
        Ent = Ent.reshape(E.size(0)*E.size(1)*E.size(2), E.size(3))
        M = torch.tanh(torch.matmul(Hinit, W['Whm']) + torch.matmul(Ent, W['Wem']))
        M = torch.matmul(M, W['Wms'])
        S = M.reshape(H.size(0)*H.size(1), H.size(2))
        S = self.softmax(S)
        S_diag = self.matrix_diag(S)
        Ent = E.clone()
        Ent = Ent.reshape(H.size(0)*H.size(1), H.size(2), H.size(3))
        attention = torch.matmul(S_diag, Ent)
        attention = attention.reshape(H.size(0), H.size(1), H.size(2), H.size(3))
        attention = attention.mean(2)
        return attention

    def attentive_combine(self, question: torch.Tensor, answer: torch.Tensor, question_kb: torch.Tensor, answer_kb: torch.Tensor):
        dim1 = question.size(2)
        dim2 = question_kb.size(2)
        transform_question = torch.einsum('ijk,kl->ijl', question, self.U1)
        attention_qa = torch.tanh(torch.matmul(transform_question, answer.permute(0, 2, 1)))
        row_max1 = self.softmax(torch.max(attention_qa, 1)[0]).unsqueeze(-1)
        column_max1 = self.softmax(torch.max(attention_qa, 2)[0]).unsqueeze(-1)

        transform_question_kb = torch.einsum('ijk,kl->ijl', question_kb, self.U2)
        attention_qa = torch.tanh(torch.matmul(transform_question_kb, answer_kb.permute(0, 2, 1)))
        row_max2 = self.softmax(torch.max(attention_qa, 1)[0]).unsqueeze(-1)
        column_max2 = self.softmax(torch.max(attention_qa, 2)[0]).unsqueeze(-1)

        row_max = torch.tanh(row_max1 + row_max2)
        column_max = torch.tanh(column_max1 + column_max2)

        question_out = torch.matmul(question.permute(0, 2, 1), column_max).reshape(-1, question.size(2))
        answer_out = torch.matmul(answer.permute(0, 2, 1), row_max).reshape(-1, answer.size(2))
        question_kb_out = torch.matmul(question_kb.permute(0, 2 ,1), column_max).reshape(-1, question_kb.size(2))
        answer_kb_out = torch.matmul(answer_kb.permute(0, 2, 1), row_max).reshape(-1, answer_kb.size(2))

        question_output = torch.cat([question_out, question_kb_out], 1)
        answer_output = torch.cat([answer_out, answer_kb_out], 1)

        return question_output, answer_output
        exit()

    def forward(self,
                question: Dict[str, torch.Tensor],
                answer: Dict[str, torch.Tensor],
                question_label: [],
                answer_label: [],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        question_embeddings = self.word_embeddings(question)
        answer_embeddings = self.word_embeddings(answer)
        question_len = question_embeddings.size(1)
        answer_len = answer_embeddings.size(1)
        batch_size = question_embeddings.size(0)
        question_pad = torch.nn.ConstantPad2d((0, 0, 0, math.ceil(self.sentence_max_length - question_len)), 0)
        answer_pad = torch.nn.ConstantPad2d((0, 0, 0, math.ceil(self.sentence_max_length - answer_len)), 0)
        question_embeddings = question_pad(question_embeddings)
        answer_embeddings = answer_pad(answer_embeddings)
        if question_len < self.sentence_max_length:
            question_mask = torch.LongTensor(np.concatenate(
                (np.ones([batch_size, question_len]), np.zeros([batch_size, self.sentence_max_length - question_len])),
                1))
        else:
            question_mask = torch.LongTensor(np.ones([batch_size, self.sentence_max_length]))
        if answer_len < self.sentence_max_length:
            answer_mask = torch.LongTensor(np.concatenate(
                (np.ones([batch_size, answer_len]), np.zeros([batch_size, self.sentence_max_length - answer_len])), 1))
        else:
            answer_mask = torch.LongTensor(np.ones([batch_size, self.sentence_max_length]))
        # question_mask = torch.LongTensor(np.concatenate((np.ones([batch_size, question_len]), np.zeros([batch_size, self.sentence_max_length - question_len])), 1))
        # answer_mask = torch.LongTensor(np.concatenate((np.ones([batch_size, answer_len]), np.zeros([batch_size, self.sentence_max_length - answer_len])), 1))
        # exit()

        # q_n, q_l = question_embeddings.size(0), question_embeddings.size(1)
        # a_n, a_l = answer_embeddings.size(0), answer_embeddings.size(1)
        #
        # question_embeddings = question_embeddings.reshape(q_n*q_l, question_embeddings.size(2))
        # answer_embeddings = answer_embeddings.reshape(a_n*a_l, answer_embeddings.size(2))
        #
        # question_embeddings = self.hidden_layer(question_embeddings)
        # answer_embeddings = self.hidden_layer(answer_embeddings)
        # question_embeddings = question_embeddings.reshape(q_n, q_l, self.embeddings_feature)
        # answer_embeddings = answer_embeddings.reshape(q_n, q_l, self.embeddings_feature)
        if torch.cuda.is_available():
            question_embeddings = question_embeddings.cuda()
            answer_embeddings = answer_embeddings.cuda()
            question_mask = question_mask.cuda()
            answer_mask = answer_mask.cuda()
        try:
            question_encode = self.encoder(question_embeddings, question_mask)[0]
            answer_encode = self.encoder(answer_embeddings, answer_mask)[0]
        except:
            print(question_embeddings.size())
            print(answer_embeddings.size())
            print(question_mask.size())
            print(answer_mask.size())
            exit()

        question_embeddings = self.hidden_layer(question_encode)
        answer_embeddings = self.hidden_layer(answer_encode)
        question_embeddings = question_embeddings.reshape(batch_size, self.sentence_max_length, self.embeddings_feature)
        answer_embeddings = answer_embeddings.reshape(batch_size, self.sentence_max_length, self.embeddings_feature)

        # type_ids = sentence['tokens-type-ids']
        # shift = [id.tolist().index(1) for id in type_ids]
        # question_count = [id.tolist().index(1)-2 for id in type_ids]
        # answer_count = [id.tolist().count(1)-1 for id in type_ids]
        # question_embedding = []
        # answer_embedding = []
        # for i, embedding in enumerate(embeddings):
        #     question_pad = torch.nn.ConstantPad2d((0, 0, 0, math.ceil(self.sentence_max_length-question_count[i])), 0)
        #     answer_pad = torch.nn.ConstantPad2d((0, 0, 0, math.ceil(self.sentence_max_length - answer_count[i])), 0)
        #     question_embedding.append(question_pad(embedding[1:shift[i]-1, :]).tolist())
        #     answer_embedding.append(answer_pad(embedding[shift[i]:answer_count[i]+shift[i]]).tolist())

        # cut = torch.nn.ConstantPad2d((0, 0, 0, -210), 0)

        question_embedding = torch.zeros(question_embeddings.size()) # question_embeddings.normal_()
        answer_embedding = torch.zeros(answer_embeddings.size()) # answer_embeddings.normal_()

        question_embedding_t = question_embedding.unsqueeze(2).repeat(1, 1, 5, 1)
        answer_embedding_t = answer_embedding.unsqueeze(2).repeat(1, 1, 5, 1)
        question_label_embedding = self.get_label_embedding(question_label)
        answer_label_embedding = self.get_label_embedding(answer_label)
        question_label_embedding = question_label_embedding.normal_()
        answer_label_embedding = answer_label_embedding.normal_()

        if torch.cuda.is_available():
            question_embedding = question_embedding.cuda()
            answer_embedding = answer_embedding.cuda()
            question_embedding_t = question_embedding_t.cuda()
            answer_embedding_t = answer_embedding_t.cuda()
            question_label_embedding = question_label_embedding.cuda()
            answer_label_embedding = answer_label_embedding.cuda()


        question_kb_embedding= self.kb_module(question_embedding_t, question_label_embedding, self.W)
        answer_kb_embedding = self.kb_module(answer_embedding_t, answer_label_embedding, self.W)
        question_conv = []
        answer_conv = []
        question_conv.append(torch.tanh(self.cnn1(question_kb_embedding)))
        question_conv.append(torch.tanh(self.cnn2(question_kb_embedding)))
        answer_conv.append(torch.tanh(self.cnn1(answer_kb_embedding)))
        answer_conv.append(torch.tanh(self.cnn2(answer_kb_embedding)))

        question_kb = torch.cat(question_conv, 2)
        answer_kb = torch.cat(answer_conv, 2)

        if torch.cuda.is_available():
            question_kb = question_kb.cuda()
            question_kb = question_kb.cuda()

        question_output, answer_output = self.attentive_combine(question_embedding, answer_embedding, question_kb, answer_kb)
        # cut = torch.nn.ConstantPad2d((0, -180, 0, 0), 0)
        # question_output = cut(question_output)
        # answer_output = cut(answer_output)
        # pad = torch.nn.ConstantPad2d((0, 180, 0, 0), 0)
        # question_output = pad(question_output)
        # answer_output = pad(answer_output)
        question_output = self.sim_layer(question_output)
        sims = torch.sum(torch.mul(question_output, answer_output), 1).unsqueeze(1)
        cat_input = torch.cat([question_output, sims, answer_output], 1)
        cat_input = self.out_layer(cat_input)
        # print(cat_input)
        cat_input = cat_input + (torch.randn(cat_input.size())*300).cuda()
        cat_input = cat_input + (torch.randn(cat_input.size())*300).cuda()

        # print(cat_input)
        # exit()
        logits = torch.nn.functional.softmax(self._classification_layer(cat_input), dim=-1)
        # logits = torch.rand(logits.size()).cuda()
        self.accuracy(logits, labels)
        self.f1(logits, labels)
        output = {"logits": logits}
        output["loss"] = self._loss(logits.view(-1, 2), labels.view(-1))
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1.get_metric(reset)
        return {"accuracy": self.accuracy.get_metric(reset), "precision": precision, "recall": recall, "f1_measure": f1_measure}