import argparse
import json

def preprocess_for_newdata(datapath, new_datas):
    trains = []
    with open(datapath + '/train_triples_rev.txt', 'r') as fin:
        for line in fin:
            trains.append(line.strip())

    news = []
    if 'FB15K' in datapath:
        rel_num = 237
    elif 'NELL23K' in datapath:
        rel_num = 200
    elif 'NELL995' in datapath:
        rel_num = 198
    for line in new_datas:
        triplet = line.strip().split()
        if triplet[0] != triplet[2]:
            if int(triplet[1].strip('R')) >=rel_num:
                temp_line = triplet[2] + '\t' + 'R'+str(int(triplet[1].strip('R'))-rel_num) + '\t' + triplet[0]
                news.append(temp_line)
            else:
                news.append(line.strip())

    adds = []

    for new in news:
        if new not in trains and new not in adds:
            adds.append(new)

    id2entity = {}
    with open(datapath + '/entity2id.txt', 'r') as fin:
        for line in fin:
            if len(line.strip().split('\t')) > 1:
                entity = line.strip().split('\t')[0]
                index = line.strip().split('\t')[1]
                id2entity[index] = entity

    id2relation = {}
    with open(datapath + '/relation2id.txt', 'r') as fin:
        for line in fin:
            if len(line.strip().split('\t')) > 1:
                relation = line.strip().split('\t')[0]
                index = line.strip().split('\t')[1]
                id2relation[index] = relation

    train_add = []
    for add in adds:
        head = add.strip().split('\t')[0]
        rel = add.strip().split('\t')[1].strip('R')
        tail = add.strip().split('\t')[2]
        try:
            train_add.append(id2entity[head] + '\t' + id2relation[rel] + '\t' + id2entity[tail])
        except:
            continue

    with open(datapath + '/train_add.txt', 'w') as fin:
        for line in train_add:
            fin.write(line+ '\n')

    train_new = []
    with open(datapath + '/train.txt', 'r') as fin:
        for line in fin:
            train_new.append(line.strip())

    for train in train_add:
        if train not in train_new:
            train_new.append(train.strip())

    with open(datapath + '/train_new.txt', 'w') as fin:
        for line in train_new:
            fin.write(line+ '\n')

def filter_threshold(datapath, a):
    with open(datapath + '/new_dict.json', 'r') as f:
        new_dict = json.load(f)
    new_datas = []
    for key,value in new_dict.items():
        if value >= a:
            new_datas.append(key)
    return new_datas

def set_pre_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='data/NELL23K', type=str)
    parser.add_argument('--a', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_pre_args()
    new_datas = filter_threshold(args.datapath, args.a)
    preprocess_for_newdata(args.datapath, new_datas)

