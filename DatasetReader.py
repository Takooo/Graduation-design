#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/20
# @Author  : Takoo
# @File    : DatasetReader.py
# @Software: PyCharm

from typing import Iterator, List, Dict, Iterable
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common import Tqdm
@DatasetReader.register("bert_know_reader")
class MyDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 max_pieces: int = 40,
                 ent_embed_size: int = 64) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.max_pieces = max_pieces
        self.ent_embed_size = ent_embed_size

    def getEntData(self, label_file_path: str):
        vec_dict = {}
        with open(label_file_path) as f:
            for line in f:
                l = line.lower().strip().split("\t")
                vec = [list(map(int, x.split(" "))) for x in l[1:]]
                for i in range(self.max_pieces - len(vec)):
                    vec.append([0] * len(vec[0]))
                vec_dict[l[0]] = vec[:self.max_pieces]
        return vec_dict

    def getEntityEmbeddings(self, ent_embedding_file_path: str):
        emb_matrix = [[0] * self.ent_embed_size]
        with open(ent_embedding_file_path, 'rb') as f:
            for line in f:
                line = line.strip().split()
                emb_matrix.append(list(map(float, line[1:])))
        return emb_matrix

    def text_to_instance(self, question_meta, question: List[Token], answer_meta, answer: List[Token], label: int = None) -> Instance:
        tokens = [Token('[CLS]')]
        tokens = tokens + question
        tokens.append(Token('[SEP]'))
        tokens = tokens + answer
        tokens.append(Token('[SEP]'))
        sentence_field = TextField(tokens, self.token_indexers)
        question_label_field = MetadataField(question_meta)
        answer_label_field = MetadataField(answer_meta)

        fields = {"sentence": sentence_field, "question_label": question_label_field, "answer_label": answer_label_field}


        if label is not None:
            label_field = LabelField(label=label, skip_indexing=True)
            fields["labels"] = label_field

        return Instance(fields)

    def read_f(self, file_path: str, label_file_path: str) -> Iterable[Instance]:
        instances = self._read(file_path, label_file_path)
        if not isinstance(instances, list):
            instances = [instance for instance in Tqdm.tqdm(instances)]
        return instances

    def _read(self, file_path: str, label_file_path: str) -> Iterator[Instance]:
        vec_dict = self.getEntData(label_file_path)
        with open(file_path) as f:
            for line in f:
                questions, answers, label = line.strip().split('\t')
                q = questions.lower()
                a = answers.lower()
                questions = questions.split()
                answers = answers.split()
                shift = len(questions) + 2
                yield self.text_to_instance(vec_dict[q], [Token(question) for question in questions],
                                            vec_dict[a], [Token(answer) for answer in answers],
                                            int(label))

