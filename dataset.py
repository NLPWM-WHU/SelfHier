import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import random


class Seq2SeqDataset(Dataset):
    def __init__(self, data_path="FB15K237/", device="cpu", args=None):
        self.data_path = data_path
        self.device = device
        self.args = args

        #entity data and vocab
        self.src_entity_file = os.path.join(data_path, "in_" + args.trainset + ".txt")  # encoder输入
        self.tgt_entity_file = os.path.join(data_path, "out_" + args.trainset + ".txt")
        with open(self.src_entity_file) as fsrc, open(self.tgt_entity_file) as ftgt:
            self.src_entity_lines = fsrc.readlines()
            self.tgt_entity_lines = ftgt.readlines()
        assert len(self.src_entity_lines) == len(self.tgt_entity_lines)
        self.entity_vocab_file = data_path + 'vocab_entity.txt'
        try:
            self.entity_dictionary = Dictionary.load(self.entity_vocab_file)
        except FileNotFoundError:
            self.entity_dictionary = Dictionary()
            self._init_entity_vocab()

        # cluster data and vocab
        self.src_cluster_file = os.path.join(data_path, "in_" + args.trainset + '_cluster' + args.K + ".txt")  # encoder输入
        self.tgt_cluster_file = os.path.join(data_path, "out_" + args.trainset + '_cluster' +  args.K + ".txt")
        with open(self.src_cluster_file) as fsrc, open(self.tgt_cluster_file) as ftgt:
            self.src_cluster_lines = fsrc.readlines()
            self.tgt_cluster_lines = ftgt.readlines()
        assert len(self.src_cluster_lines) == len(self.tgt_cluster_lines)
        self.cluster_vocab_file = data_path + 'vocab_cluster' + args.K +'.txt'
        try:
            self.cluster_dictionary = Dictionary.load(self.cluster_vocab_file)
        except FileNotFoundError:
            self.cluster_dictionary = Dictionary()
            self._init_cluster_vocab()

        # relation data and vocab
        self.src_relation_file = os.path.join(data_path, "in_" + args.trainset + ".txt")  # encoder输入
        self.tgt_relation_file = os.path.join(data_path, "out_" + args.trainset + '_relation.txt')
        with open(self.src_relation_file) as fsrc, open(self.tgt_relation_file) as ftgt:
            self.src_relation_lines = fsrc.readlines()
            self.tgt_relation_lines = ftgt.readlines()
        assert len(self.src_relation_lines) == len(self.tgt_relation_lines)
        self.relation_vocab_file = data_path + 'vocab_relation.txt'
        try:
            self.relation_dictionary = Dictionary.load(self.relation_vocab_file)
        except FileNotFoundError:
            self.relation_dictionary = Dictionary()
            self._init_relation_vocab()

        self.padding_idx = self.entity_dictionary.pad()
        self.len_entity_vocab = len(self.entity_vocab_file)
        self.len_cluster_vocab = len(self.cluster_vocab_file)
        self.len_relation_vocab = len(self.relation_vocab_file)
        self.smart_filter = args.smart_filter
        self.args = args

    def __len__(self):
        return len(self.src_entity_lines)

    def _init_entity_vocab(self):
        self.entity_dictionary.add_symbol('LOOP')
        N = -1
        with open(self.data_path + 'relation2id.txt') as fin:
            for line in fin:
                N += 1
        rev_r = []
        with open(self.data_path + 'relation2id.txt') as fin:
            for line in fin:
                try:
                    r, rid = line.strip().split('\t')
                    rev_rid = int(rid) + N
                    self.entity_dictionary.add_symbol('R' + rid)
                    rev_r.append('R' + str(rev_rid))
                except:
                    continue
        for r in rev_r:
            self.entity_dictionary.add_symbol(r)
        with open(self.data_path + 'entity2id.txt') as fin:
            for line in fin:
                try:
                    e, eid = line.strip().split('\t')
                    self.entity_dictionary.add_symbol(eid)
                except:
                    continue
        self.entity_dictionary.save(self.entity_vocab_file)

    def _init_cluster_vocab(self):
        self.cluster_dictionary.add_symbol('LOOP')
        N = -1
        with open(self.data_path + 'relation2id.txt') as fin:
            for line in fin:
                N += 1
        rev_r = []
        with open(self.data_path + 'relation2id.txt') as fin:
            for line in fin:
                try:
                    r, rid = line.strip().split('\t')
                    rev_rid = int(rid) + N
                    self.cluster_dictionary.add_symbol('R' + rid)
                    rev_r.append('R' + str(rev_rid))
                except:
                    continue
        for r in rev_r:
            self.cluster_dictionary.add_symbol(r)

        with open(self.data_path + 'cluster2id' + self.args.K + '.txt') as fin:
            for line in fin:
                c, cid = line.strip().split('\t')
                self.cluster_dictionary.add_symbol(c)

        self.cluster_dictionary.save(self.cluster_vocab_file)

    def _init_relation_vocab(self):
        self.relation_dictionary.add_symbol('LOOP')
        N = -1
        with open(self.data_path + 'relation2id.txt') as fin:
            for line in fin:
                N += 1
        rev_r = []
        with open(self.data_path + 'relation2id.txt') as fin:
            for line in fin:
                try:
                    r, rid = line.strip().split('\t')
                    rev_rid = int(rid) + N
                    self.relation_dictionary.add_symbol('R' + rid)
                    rev_r.append('R' + str(rev_rid))
                except:
                    continue
        for r in rev_r:
            self.relation_dictionary.add_symbol(r)
        with open(self.data_path + 'multi_relation2id.txt') as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                self.relation_dictionary.add_symbol(r)
        with open(self.data_path + 'entity2id.txt') as fin:
            for line in fin:
                try:
                    e, eid = line.strip().split('\t')
                    self.relation_dictionary.add_symbol(eid)
                except:
                    continue
        self.relation_dictionary.save(self.relation_vocab_file)

    def __getitem__(self, index):
        src_entity_line = self.src_entity_lines[index].strip().split(' ')
        tgt_entity_line = self.tgt_entity_lines[index].strip().split(' ')
        source_entity_id = self.entity_dictionary.encode_line(src_entity_line)
        target_entity_id = self.entity_dictionary.encode_line(tgt_entity_line)
        src_cluster_line = self.src_cluster_lines[index].strip().split(' ')
        tgt_cluster_line = self.tgt_cluster_lines[index].strip().split(' ')
        source_cluster_id = self.cluster_dictionary.encode_line(src_cluster_line)
        target_cluster_id = self.cluster_dictionary.encode_line(tgt_cluster_line)
        src_relation_line = self.src_relation_lines[index].strip().split(' ')
        tgt_relation_line = self.tgt_relation_lines[index].strip().split(' ')
        source_relation_id = self.relation_dictionary.encode_line(src_relation_line)
        target_relation_id = self.relation_dictionary.encode_line(tgt_relation_line)

        l_entity = len(target_entity_id)
        mask_entity = torch.ones_like(target_entity_id)
        if self.args.mask:
            for i in range(0, l_entity - 2):
                if i % 2 == 0:  # do not mask relation
                    continue
                if random.random() < self.args.prob:  # randomly replace with prob
                    target_entity_id[i] = random.randint(0, self.len_entity_vocab - 1)
                    mask_entity[i] = 0

        l_cluster = len(target_cluster_id)
        mask_cluster = torch.ones_like(target_cluster_id)
        if self.args.mask:
            for i in range(0, l_cluster - 2):
                if i % 2 == 0:  # do not mask relation
                    continue
                if random.random() < self.args.prob:  # randomly replace with prob
                    target_cluster_id[i] = random.randint(0, self.len_cluster_vocab - 1)
                    mask_cluster[i] = 0

        l_relation = len(target_relation_id)
        mask_relation = torch.ones_like(target_relation_id)
        if self.args.mask:
            for i in range(0, l_relation - 2):
                if i % 2 == 0:  # do not mask relation
                    continue
                if random.random() < self.args.prob:  # randomly replace with prob
                    target_relation_id[i] = random.randint(0, self.len_relation_vocab - 1)
                    mask_relation[i] = 0

        return {
            "id": index,
            "tgt_length": len(target_entity_id),
            "source_entity": source_entity_id,
            "target_entity": target_entity_id,
            "mask_entity": mask_entity,
            "source_cluster": source_cluster_id,
            "target_cluster": target_cluster_id,
            "mask_cluster": mask_cluster,
            "source_relation": source_relation_id,
            "target_relation": target_relation_id,
            "mask_relation": mask_relation,
        }

    def collate_fn(self, samples):
        lens = [sample["tgt_length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens) #batch_size

        source_entity = torch.LongTensor(bsz, 3)
        source_cluster = torch.LongTensor(bsz, 3)
        source_relation = torch.LongTensor(bsz, 3)

        prev_outputs_entity = torch.LongTensor(bsz, max_len)
        mask_entity = torch.zeros(bsz, max_len)
        prev_outputs_cluster = torch.LongTensor(bsz, max_len)
        mask_cluster = torch.zeros(bsz, max_len)
        prev_outputs_relation = torch.LongTensor(bsz, max_len)
        mask_relation = torch.zeros(bsz, max_len)

        source_entity[:, 0].fill_(self.entity_dictionary.bos())
        prev_outputs_entity.fill_(self.entity_dictionary.pad())
        prev_outputs_entity[:, 0].fill_(self.entity_dictionary.bos())
        target_entity = copy.deepcopy(prev_outputs_entity)

        source_cluster[:, 0].fill_(self.cluster_dictionary.bos())
        prev_outputs_cluster.fill_(self.cluster_dictionary.pad())
        prev_outputs_cluster[:, 0].fill_(self.cluster_dictionary.bos())
        target_cluster = copy.deepcopy(prev_outputs_cluster)

        source_relation[:, 0].fill_(self.relation_dictionary.bos())
        prev_outputs_relation.fill_(self.relation_dictionary.pad())
        prev_outputs_relation[:, 0].fill_(self.relation_dictionary.bos())
        target_relation = copy.deepcopy(prev_outputs_relation)

        ids = []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])

            source_entity_ids = sample["source_entity"]
            target_entity_ids = sample["target_entity"]
            source_entity[idx, 1:] = source_entity_ids[: -1]
            prev_outputs_entity[idx, 1:sample["tgt_length"]] = target_entity_ids[: -1]
            target_entity[idx, 0: sample["tgt_length"]] = target_entity_ids
            mask_entity[idx, 0: sample["tgt_length"]] = sample["mask_entity"]

            source_cluster_ids = sample["source_cluster"]
            target_cluster_ids = sample["target_cluster"]
            source_cluster[idx, 1:] = source_cluster_ids[: -1]
            prev_outputs_cluster[idx, 1:sample["tgt_length"]] = target_cluster_ids[: -1]
            target_cluster[idx, 0: sample["tgt_length"]] = target_cluster_ids
            mask_cluster[idx, 0: sample["tgt_length"]] = sample["mask_cluster"]

            source_relation_ids = sample["source_relation"]
            target_relation_ids = sample["target_relation"]
            source_relation[idx, 1:] = source_relation_ids[: -1]
            try:
                prev_outputs_relation[idx, 1:sample["tgt_length"]] = target_relation_ids[: -1]
            except:
                print("111111")
            target_relation[idx, 0: sample["tgt_length"]] = target_relation_ids
            mask_relation[idx, 0: sample["tgt_length"]] = sample["mask_relation"]

        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "source_entity": source_entity.to(self.device),
            "prev_outputs_entity": prev_outputs_entity.to(self.device),
            "target_entity": target_entity.to(self.device),
            "mask_entity": mask_entity.to(self.device),
            "source_cluster": source_cluster.to(self.device),
            "prev_outputs_cluster": prev_outputs_cluster.to(self.device),
            "target_cluster": target_cluster.to(self.device),
            "mask_cluster": mask_cluster.to(self.device),
            "source_relation": source_relation.to(self.device),
            "prev_outputs_relation": prev_outputs_relation.to(self.device),
            "target_relation": target_relation.to(self.device),
            "mask_relation": mask_relation.to(self.device),
        }

    def get_next_valid(self):
        train_valid = dict()
        eval_valid = dict()
        vocab_size = len(self.entity_dictionary)
        eos = self.entity_dictionary.eos()
        with open(self.data_path + 'train_triples_rev.txt', 'r') as f:
             for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.entity_dictionary.indices[h] #头实体id
                rid = self.entity_dictionary.indices[r] #关系id
                tid = self.entity_dictionary.indices[t] #尾实体id
                e = hid
                er = vocab_size * rid + hid #词表大小 * 关系id + 头实体id
                if e not in train_valid:
                    if self.smart_filter:
                        train_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[e] = [eos, ]
                if er not in train_valid:
                    if self.smart_filter:
                        train_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[er] = []
                if self.smart_filter:
                    train_valid[e][rid] = 0
                    train_valid[e][eos] = 0
                    train_valid[er][tid] = 0
                else:
                    train_valid[e].append(rid)
                    train_valid[er].append(tid)

        with open(self.data_path + 'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.entity_dictionary.indices[h]
                rid = self.entity_dictionary.indices[r]
                tid = self.entity_dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in eval_valid:
                    if self.smart_filter:
                        eval_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[e] = [eos, ]
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[e][rid] = 0
                    eval_valid[e][eos] = 0
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[e].append(rid)
                    eval_valid[er].append(tid)

        with open(self.data_path + 'valid_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.entity_dictionary.indices[h]
                rid = self.entity_dictionary.indices[r]
                tid = self.entity_dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)

        with open(self.data_path + 'test_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.entity_dictionary.indices[h]
                rid = self.entity_dictionary.indices[r]
                tid = self.entity_dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)

        return train_valid, eval_valid


class TestDataset(Dataset):
    def __init__(self, data_path="FB15K237/", device="cpu", src_file=None, args=None):
        if src_file:
            self.src_file = os.path.join(data_path, src_file)
        else:
            self.src_file = os.path.join(data_path, "valid_triples.txt")

        with open(self.src_file) as f:
            self.src_lines = f.readlines()

        self.vocab_file = data_path + 'vocab_entity.txt'
        self.device = device

        try:
            self.dictionary = Dictionary.load(self.vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()

        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.args = args

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):

        src_line = self.src_lines[index].strip().split('\t')
        source_id = self.dictionary.encode_line(src_line[:2])
        target_id = self.dictionary.encode_line(src_line[2:])
        return {
            "id": index,
            "source": source_id,
            "target": target_id,
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        source = torch.LongTensor(bsz, 3)
        target = torch.LongTensor(bsz, 1)

        source[:, 0].fill_(self.dictionary.bos())

        ids = []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]
            source[idx, 1:] = source_ids[: -1]
            target[idx, 0] = target_ids[: -1]

        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "source": source.to(self.device), # bos + c + h + r
            "target": target.to(self.device) # t
        }