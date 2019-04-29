#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18
# @Author  : Takoo
# @File    : allennlp_bert.py
# @Software: PyCharm

from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import BertEmbedder, PretrainedBertEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def getEntityEmbeddings(self, file_path: str):
    emb_matrix = [[0] * 64]
    with open(file_path, "rb") as f:
        for line in f:
            line = line.strip()
            items = line.split()
            emb_matrix.append(map(float, items[1:]))
    return emb_matrix

class MyDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
    def text_to_instance(self, question: List[Token], answer: List[Token], label: int = None) -> Instance:
        tokens = [Token('[CLS]')]
        tokens = tokens + question
        tokens.append(Token('[SEP]'))
        tokens = tokens + answer
        tokens.append(Token('[SEP]'))
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if label is not None:
            label_field = LabelField(label=label, skip_indexing = True)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str, label_file_path: str) -> Iterator[Instance]:
        vec_dict = {}
        with open(file_path) as f, open(label_file_path) as fl:
            for line in fl:
                l = line.lower().strip().split("\t")
                vec = [map(int, x.split(" ")) for x in l[1:]]
                for i in range(512 - len(vec)):
                    vec.append([0] * len(vec[0]))
                vec_dict[l[0]] = vec[:512]

            for line in f:
                questions, answers, label = line.strip().split('\t')
                questions = questions.split()
                answers = answers.split()
                yield self.text_to_instance([Token(question) for question in questions],
                                            [Token(answer) for answer in answers],
                                            int(label))

class BasicClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self._seq2vec_encoder = seq2vec_encoder
        self._classification_layer = torch.nn.Linear(word_embeddings.get_output_dim(),
                                          out_features=2)

        self.accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        embeddings = embeddings[:,0,:]
        logits = self._classification_layer(embeddings)
        # probs = torch.nn.functional.sigmoid(logits)
        probs = torch.nn.functional.log_softmax(logits, dim=1)
        print(probs.view(-1, 2).size(0))
        print(labels.view(-1).size())
        exit()
        output = {"logits": logits, "probs": probs}
        output["loss"] = self._loss(logits.view(-1, 2), labels.view(-1))
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

reader = MyDatasetReader(token_indexers={"tokens": PretrainedBertIndexer('/Users/takoo/Public/pretrained/bert-base-uncased')})
train_dataset = reader.read('/Users/takoo/Public/tmp/data/wikiQA-train.txt')
validation_dataset = reader.read('/Users/takoo/Public/tmp/data/wikiQA-test.txt')
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = PretrainedBertEmbedder(pretrained_model="/Users/takoo/Public/pretrained/bert-base-uncased")
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}, allow_unmatched_keys=True)

cnn_encoder = CnnEncoder(word_embeddings.get_output_dim(), num_filters=100,
                         ngram_filter_sizes=(2, 3, 4, 5), output_dim=256)


# lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = BasicClassifier(word_embeddings, cnn_encoder, vocab)
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
                  num_epochs=1000,
                  cuda_device=cuda_device)
trainer.train()


# predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
# tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
# tag_ids = np.argmax(tag_logits, axis=-1)
# print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
# # Here's how to save the model.
# with open("/tmp/model.th", 'wb') as f:
#     torch.save(model.state_dict(), f)
# vocab.save_to_files("/tmp/vocabulary")
# # And here's how to reload the model.
# vocab2 = Vocabulary.from_files("/tmp/vocabulary")
# model2 = LstmTagger(word_embeddings, vocab2)
# with open("/tmp/model.th", 'rb') as f:
#     model2.load_state_dict(torch.load(f))
# if cuda_device > -1:
#     model2.cuda(cuda_device)
# predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
# tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
# np.testing.assert_array_almost_equal(tag_logits2, tag_logits)