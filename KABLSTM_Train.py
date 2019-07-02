#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/1
# @Author  : Takoo
# @File    : KABLSTM_Train.py
# @Software: PyCharm

import torch
import torch.optim as optim
import configparser
import os
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate
from allennlp.common.util import dump_metrics
from KABLSTM_DatasetReader import MyDatasetReader
from KABLSTM_Model import BasicClassifier


config = configparser.ConfigParser()
config.read('./KABLSTM_config.json')

torch.manual_seed(1)
# BERT_BASE_UNCASED = config.get('local_file', 'BERT_BASE_UNCASED')
# WIKIQA_TRAIN = config.get('local_file', 'WIKIQA_TRAIN')
# WIKIQA_TEST = config.get('local_file', 'WIKIQA_TEST')
# WIKIQA_TRAIN_LABEL = config.get('local_file', 'WIKIQA_TRAIN_LABEL')
# WIKIQA_TEST_LABEL = config.get('local_file', 'WIKIQA_TEST_LABEL')
# WIKIQA_KB_EMBED = config.get('local_file', 'WIKIQA_KB_EMBED')

GLOVE_840B_300D = config.get('server_file', 'GLOVE_840B_300D')
# WIKIQA_TRAIN = config.get('server_file', 'WIKIQA_TRAIN')
# WIKIQA_DEV = config.get('server_file', "WIKIQA_DEV")
# WIKIQA_TEST = config.get('server_file', 'WIKIQA_TEST')
# WIKIQA_TRAIN_LABEL = config.get('server_file', 'WIKIQA_TRAIN_LABEL')
# WIKIQA_DEV_LABEL = config.get('server_file', "WIKIQA_DEV_LABEL")
# WIKIQA_TEST_LABEL = config.get('server_file', 'WIKIQA_TEST_LABEL')
# WIKIQA_KB_EMBED = config.get('server_file', 'WIKIQA_KB_EMBED')

TREC_TRAIN = config.get('server_file', 'TREC_TRAIN')
TREC_DEV = config.get('server_file', "TREC_DEV")
TREC_TEST = config.get('server_file', 'TREC_TEST')
TREC_TRAIN_LABEL = config.get('server_file', 'TREC_TRAIN_LABEL')
TREC_DEV_LABEL = config.get('server_file', "TREC_DEV_LABEL")
TREC_TEST_LABEL = config.get('server_file', 'TREC_TEST_LABEL')
TREC_KB_EMBED = config.get('server_file', 'TREC_KB_EMBED')




reader = MyDatasetReader()
train_dataset = reader.read_f(TREC_TRAIN, TREC_TRAIN_LABEL)
dev_dataset = reader.read_f(TREC_DEV, TREC_DEV_LABEL)
test_dataset = reader.read_f(TREC_TEST, TREC_TEST_LABEL)
ent_embeddings = reader.getEntityEmbeddings(TREC_KB_EMBED)
vocab = Vocabulary.from_instances(train_dataset + dev_dataset + test_dataset)

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
                  validation_dataset=dev_dataset,
                  patience=int(config.get('Train', 'patience')),
                  num_epochs=int(config.get('Train', 'num_epochs')),
                  serialization_dir=config.get('Train', 'trecqa_serialization_dir'),
                  cuda_device=cuda_device
                  )

metrics = trainer.train()
print("train ended")
test_metrics = evaluate(trainer.model, test_dataset, iterator,
                                cuda_device=trainer._cuda_devices[0], # pylint: disable=protected-access,
                                batch_weight_key="")
for key, value in test_metrics.items():
    metrics["test_" + key] = value

dump_metrics(os.path.join(config.get('Train', 'trecqa_serialization_dir'), "KABLSTM_metrics.json"), metrics, log=True)