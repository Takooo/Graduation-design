#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/20
# @Author  : Takoo
# @File    : Train.py
# @Software: PyCharm

import torch
import torch.optim as optim
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from DatasetReader import MyDatasetReader
from Model import BasicClassifier

torch.manual_seed(1)
# BERT_BASE_UNCASED = '/Users/takoo/Public/pretrained/bert-base-uncased'
# WIKIQA_TRAIN = '/Users/takoo/Public/tmp/data/wikiQA-train.txt'
# WIKIQA_TEST = '/Users/takoo/Public/tmp/data/wikiQA-test.txt'
# WIKIQA_TRAIN_LABEL = '/Users/takoo/Public/tmp/data/wikiQA-train.txt.labeled'
# WIKIQA_TEST_LABEL = '/Users/takoo/Public/tmp/data/wikiQA-test.txt.labeled'
# WIKIQA_KB_EMBED = '/Users/takoo/Public/tmp/embed/fb5m-wiki.transE'

BERT_BASE_UNCASED = '/home/zhangfan/uncased_L-12_H-768_A-12'
WIKIQA_TRAIN = '/home/zhangfan/data/wikiQA-train.txt'
WIKIQA_TEST = '/home/zhangfan/data/wikiQA-test.txt'
WIKIQA_TRAIN_LABEL = '/home/zhangfan/data/wikiQA-train.txt.labeled'
WIKIQA_TEST_LABEL = '/home/zhangfan/data/wikiQA-test.txt.labeled'
WIKIQA_KB_EMBED = '/home/zhangfan/embed/fb5m-wiki.transE'



reader = MyDatasetReader(token_indexers={"tokens": PretrainedBertIndexer(BERT_BASE_UNCASED)})
train_dataset = reader.read_f(WIKIQA_TRAIN, WIKIQA_TRAIN_LABEL)
validation_dataset = reader.read_f(WIKIQA_TEST, WIKIQA_TEST_LABEL)
ent_embeddings = reader.getEntityEmbeddings(WIKIQA_KB_EMBED)
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = PretrainedBertEmbedder(pretrained_model=BERT_BASE_UNCASED)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}, allow_unmatched_keys=True)

cnn_encoder = CnnEncoder(word_embeddings.get_output_dim(), num_filters=100,
                         ngram_filter_sizes=(2, 3, 4, 5), output_dim=256)


model = BasicClassifier(word_embeddings, ent_embeddings, vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
optimizer = optim.Adam(model.parameters())
iterator = BasicIterator(batch_size=32)
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=10,
                  cuda_device=cuda_device
                  )
trainer.train()