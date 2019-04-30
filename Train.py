#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/20
# @Author  : Takoo
# @File    : Train.py
# @Software: PyCharm

import torch
import torch.optim as optim
import configparser
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from DatasetReader import MyDatasetReader
from Model import BasicClassifier


config = configparser.ConfigParser()
config.read('./config.json')

torch.manual_seed(1)
# BERT_BASE_UNCASED = config.get('local_file', 'BERT_BASE_UNCASED')
# WIKIQA_TRAIN = config.get('local_file', 'WIKIQA_TRAIN')
# WIKIQA_TEST = config.get('local_file', 'WIKIQA_TEST')
# WIKIQA_TRAIN_LABEL = config.get('local_file', 'WIKIQA_TRAIN_LABEL')
# WIKIQA_TEST_LABEL = config.get('local_file', 'WIKIQA_TEST_LABEL')
# WIKIQA_KB_EMBED = config.get('local_file', 'WIKIQA_KB_EMBED')

BERT_BASE_UNCASED = config.get('server_file', 'BERT_BASE_UNCASED')
WIKIQA_TRAIN = config.get('server_file', 'WIKIQA_TRAIN')
WIKIQA_TEST = config.get('server_file', 'WIKIQA_TEST')
WIKIQA_TRAIN_LABEL = config.get('server_file', 'WIKIQA_TRAIN_LABEL')
WIKIQA_TEST_LABEL = config.get('server_file', 'WIKIQA_TEST_LABEL')
WIKIQA_KB_EMBED = config.get('server_file', 'WIKIQA_KB_EMBED')

reader = MyDatasetReader(token_indexers={"tokens": PretrainedBertIndexer(BERT_BASE_UNCASED)})
train_dataset = reader.read_f(WIKIQA_TRAIN, WIKIQA_TRAIN_LABEL)
validation_dataset = reader.read_f(WIKIQA_TEST, WIKIQA_TEST_LABEL)
ent_embeddings = reader.getEntityEmbeddings(WIKIQA_KB_EMBED)
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = PretrainedBertEmbedder(pretrained_model=BERT_BASE_UNCASED)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}, allow_unmatched_keys=True)

model = BasicClassifier(word_embeddings, ent_embeddings, vocab, config)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
optimizer = optim.Adam(model.parameters())
iterator = BasicIterator(batch_size=int(config.get('iterator', 'batch_size')))
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=int(config.get('Train', 'patience')),
                  num_epochs=int(config.get('Train', 'num_epochs')),
                  cuda_device=cuda_device
                  )
trainer.train()