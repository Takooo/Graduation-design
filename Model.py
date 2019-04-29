#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/20
# @Author  : Takoo
# @File    : Model.py
# @Software: PyCharm

from typing import Dict
import torch
import numpy as np
import math
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


class BasicClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 ent_embeddings,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.ent_embeddings = ent_embeddings
        self.hidden_layer = torch.nn.Linear(word_embeddings.get_output_dim(),
                                          out_features=64)
        self.cnn1 = torch.nn.Conv1d(40, 40, 2)
        self.cnn1.bias.data.fill_(0.1)
        torch.nn.init.xavier_normal_(self.cnn1.weight)
        self.cnn2 = torch.nn.Conv1d(40, 40, 3)
        self.cnn2.bias.data.fill_(0.1)
        torch.nn.init.xavier_normal_(self.cnn2.weight)
        self.softmax = torch.nn.Softmax(-1)
        self.sim_layer = torch.nn.Linear(189, 189)
        self.out_layer = torch.nn.Linear(379, 200)
        self._classification_layer = torch.nn.Linear(200, 2)
        self.accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self.W = {
            'Whm': torch.autograd.Variable(torch.FloatTensor(64, 200).uniform_(-1, 1), requires_grad=True),
            'Wem': torch.autograd.Variable(torch.FloatTensor(64, 200).uniform_(-1, 1), requires_grad=True),
            'Wms': torch.autograd.Variable(torch.FloatTensor(200, 1).uniform_(-1, 1), requires_grad=True)
        }

        self.U1 = torch.autograd.Variable(torch.nn.init.xavier_uniform_(torch.Tensor(64, 64)), requires_grad=True)
        self.U2 = torch.autograd.Variable(torch.nn.init.xavier_uniform_(torch.Tensor(125, 125)), requires_grad=True)

        if torch.cuda.is_available():
            self.W['Whm'] = self.W['Whm'].cuda()
            self.W['Wem'] = self.W['Wem'].cuda()
            self.W['Wms'] = self.W['Wms'].cuda()
            self.U1 = self.U1.cuda()
            self.U2 = self.U2.cuda()

    def get_label_embedding(self, sentence_label):
        label_embed = np.zeros((len(sentence_label), len(sentence_label[0]), len(sentence_label[0][0]), 64))
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

        # print("--------")
        # print(question.size())
        # print(answer.size())
        # print(question_kb.size())
        # print(answer_kb.size())
        #
        # print(row_max.size())
        # print(column_max.size())
        #
        # print(question_out.size())
        # print(answer_out.size())
        # print(question_kb_out.size())
        # print(answer_kb_out.size())

        question_output = torch.cat([question_out, question_kb_out], 1)
        answer_output = torch.cat([answer_out, answer_kb_out], 1)

        # print(question_output.size())
        # print(answer_output.size())
        return question_output, answer_output
        exit()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                question_label: [],
                answer_label: [],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        n, l = embeddings.size(0), embeddings.size(1)
        embeddings = embeddings.reshape(n*l, embeddings.size(2))
        embeddings = self.hidden_layer(embeddings)
        embeddings = embeddings.reshape(n, l, 64)
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()

        type_ids = sentence['tokens-type-ids']
        shift = [id.tolist().index(1) for id in type_ids]
        question_count = [id.tolist().index(1)-2 for id in type_ids]
        answer_count = [id.tolist().count(1)-1 for id in type_ids]
        question_embedding = []
        answer_embedding = []
        for i, embedding in enumerate(embeddings):
            question_pad = torch.nn.ConstantPad2d((0, 0, 0, math.ceil(40-question_count[i])), 0)
            answer_pad = torch.nn.ConstantPad2d((0, 0, 0, math.ceil(40 - answer_count[i])), 0)
            question_embedding.append(question_pad(embedding[1:shift[i]-1, :]).tolist())
            answer_embedding.append(answer_pad(embedding[shift[i]:answer_count[i]+shift[i]]).tolist())

        question_embedding = torch.Tensor(question_embedding).normal_()
        answer_embedding = torch.Tensor(answer_embedding).normal_()

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

        # print("EEEEE")
        # print(embeddings.size())
        # print(embeddings)
        # print()
        # print("QQQQQ")
        # print(torch.Tensor(question_embedding).size())
        # print(torch.Tensor(question_embedding))
        # print("KKKKK")
        # print(question_label_embedding.size())
        # print(question_label_embedding)
        # print("NNNNN")
        # print(question_label)
        # exit()

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
        # M = torch.tanh(torch.matmul(question_kb_embedding, Wqm)+torch.matmul(question_label_embedding, Wqlm))
        question_output = self.sim_layer(question_output)
        sims = torch.sum(torch.mul(question_output, answer_output), 1).unsqueeze(1)
        cat_input = torch.cat([question_output, sims, answer_output], 1)
        cat_input = self.out_layer(cat_input)
        logits = torch.nn.functional.softmax(self._classification_layer(cat_input), dim=-1)
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        self.accuracy(logits, labels)
        # print(logits)
        # print(probs)
        # print(labels)
        # exit()
        output = {"logits": logits}
        output["loss"] = self._loss(logits.view(-1, 2), labels.view(-1))
        # print(probs)
        # exit()
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}