#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/1
# @Author  : Takoo
# @File    : KABLSTM_Train.py
# @Software: PyCharm

import torch
import torch.optim as optim
import configparser
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder,Embedding
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from KABLSTM_DatasetReader import MyDatasetReader
from KABLSTM_Model import BasicClassifier


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
GLOVE_840B_300D = config.get('server_file', 'GLOVE_840B_300D')


reader = MyDatasetReader()
train_dataset = reader.read_f(WIKIQA_TRAIN, WIKIQA_TRAIN_LABEL)
validation_dataset = reader.read_f(WIKIQA_TEST, WIKIQA_TEST_LABEL)
ent_embeddings = reader.getEntityEmbeddings(WIKIQA_KB_EMBED)
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=300,
                            pretrained_file = GLOVE_840B_300D,
                            trainable = True)
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