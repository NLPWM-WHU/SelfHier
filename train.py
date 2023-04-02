import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset, TestDataset
from model import TransformerModel
import argparse
import numpy as np
import os
import json
import random
from tqdm import tqdm
import logging
import transformers
import math
from transformers import top_k_top_p_filtering

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=256, type=int) # 512
    parser.add_argument("--hidden-size", default=512, type=int) # 512
    parser.add_argument("--num-layers", default=6, type=int) # 6
    parser.add_argument("--batch-size", default=1024, type=int) # 64
    parser.add_argument("--test-batch-size", default=16, type=int) # 16
    parser.add_argument("--lr", default=1e-4, type=float) # 1e-4
    parser.add_argument("--dropout", default=0.1, type=float) # 0.1
    parser.add_argument("--weight-decay", default=0, type=float) #1e-3
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--save-dir", default="model_1")
    parser.add_argument("--ckpt", default="ckpt-relation.pt")
    parser.add_argument("--dataset", default="FB15K237")
    parser.add_argument("--label-smooth", default=0.5, type=float) # 0.5
    parser.add_argument("--l-punish", default=False, action="store_true") # during generation, add punishment for length
    parser.add_argument("--beam-size", default=128, type=int) # during generation, beam size
    parser.add_argument("--no-filter-gen", default=True, action="store_true") # during generation, not filter unreachable next token
    parser.add_argument("--test", default=False, action="store_true") # for test mode
    parser.add_argument("--encoder", default=False, action="store_true") # only use TransformerEncoder
    parser.add_argument("--trainset", default="6_rev_rule") # FB15K237: "6_rev", "60", "30_rev", FB15K237-10: "30", "30_rev"
    parser.add_argument("--loop", default=False, action="store_true") # add self-loop instead of <eos>
    parser.add_argument("--prob", default=0, type=float) # ratio of replaced token
    parser.add_argument("--max-len", default=3, type=int) # maximum number of hops considered
    parser.add_argument("--iter", default=False, action="store_true") # switch for iterative training
    parser.add_argument("--iter-batch-size", default=128, type=int) # FB15K237, codex-m: 128; NELL995: 32
    parser.add_argument("--smart-filter", default=False, action="store_true") # more space consumed, less time; switch on when --filter-gen
    parser.add_argument("--warmup", default=3, type=float) # warmup steps ratio
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval-begin', default=0, type=int)
    parser.add_argument('--K', default='50', type=str)
    parser.add_argument('--R', default='3', type=str)
    parser.add_argument("--mask", default=True, action="store_true")
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--gmma', default=0.01, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--theta', default=0.01, type=float)
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--loss', default=2, type=int)
    parser.add_argument('--use-pretrained-emb', default=True, action="store_true")
    parser.add_argument('--xavier-all', default=False, action="store_true")
    parser.add_argument('--valid', default=False, action="store_true")
    parser.add_argument('--relation', default=True, action="store_true")
    parser.add_argument('--cluster', default=True, action="store_true")
    parser.add_argument('--eval-pattern', default=0, type=int)
    parser.add_argument("--self-consistency", default=True, action="store_true")
    parser.add_argument("--output-path", default=False, action="store_true")
    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, args, true_triples=None, valid_triples=None):
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    restricted_punish = -30
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
    vocab_size = len(model.dictionary)
    eos = model.dictionary.eos()
    bos = model.dictionary.bos()
    rev_dict = dict()
    for k in model.dictionary.indices.keys():
        v = model.dictionary.indices[k]
        rev_dict[v] = k
    pres = []
    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f" % (mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count)))
            batch_size = samples["source"].size(0)
            candidates = [dict() for i in range(batch_size)]
            candidates_path = [dict() for i in range(batch_size)]
            source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix[:, :, 0].fill_(model.dictionary.bos())
            lprob = torch.zeros([batch_size, beam_size]).to(device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token: choose beam_size from only vocab_size, initiate prefix
            tmp_source = samples["source"]
            tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix[:, 0].fill_(model.dictionary.bos())
            logits = model.logits(tmp_source, tmp_prefix).squeeze()
            logits = F.log_softmax(logits, dim=-1)
            logits = logits.view(-1, vocab_size)
            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
            prefix[:, :, 1] = argsort[:, :]
            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                all_logits = model.logits(source.view(bb, -1), prefix.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                logits = F.log_softmax(logits, dim=-1)
                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen = tmp_clen + 1
                tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                if l == max_len-1:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :(2*beam_size)]
                else:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]
                prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                clen = torch.gather(input=tmp_clen, dim=1, index=argsort)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if prefix[i][j][l].item() == eos:
                            candidate = prefix[i][j][l-1].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(l / 2)
                            else:
                                prob = lprob[i][j].item()
                            lprob[i][j] -= 10000
                            if candidate not in candidates[i]:
                                candidates[i][candidate] = prob
                                candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                candidates[i][candidate] = max(candidates[i][candidate], prob)
                # no <end> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate = prefix[i][j][l].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(max_len/2)
                            else:
                                prob = lprob[i][j].item()
                            if candidate not in candidates[i]:
                                candidates[i][candidate] = prob
                                candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()                                
                                candidates[i][candidate] = max(candidates[i][candidate], prob)
            target = samples["target"].cpu()
            for i in range(batch_size):
                hid = samples["source"][i][-2].item()
                rid = samples["source"][i][-1].item()
                index = vocab_size * rid + hid
                if index in valid_triples:
                    mask = valid_triples[index]
                    for tid in candidates[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates[i][tid] -= 100000
                count += 1
                candidate_ = sorted(zip(candidates[i].items(), candidates_path[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate = [pair[0][0] for pair in candidate_]
                candidate_path = [pair[1][1] for pair in candidate_]
                candidate = torch.from_numpy(np.array(candidate))
                ranking = (candidate[:] == target[i]).nonzero()
                if ranking.nelement() != 0:
                    path = candidate_path[ranking]
                    ranking = 1 + ranking.item()
                    pres.append(ranking)
                    mrr += (1 / ranking)
                    hit += 1
                    if ranking <= 1:
                        hit1 += 1
                    if ranking <= 3:
                        hit3 += 1
                    if ranking <= 10:
                        hit10 += 1
                else:
                    pres.append(-999999)
    with open('./data/FB15K237-20/pre' + str(args.cluster) + str(args.relation) + str(args.loss) + str(args.loss), 'w') as f:
        for pre in pres:
            f.write(str(pre) + '\n')

    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]" % (mrr/count, hit1/count, hit3/count, hit10/count))
    return hit/count, hit1/count, hit3/count, hit10/count

def evaluate1(model1, model2, dataloader, device, args, true_triples=None, valid_triples=None):
    model1.eval()
    model2.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
    vocab_size = len(model1.dictionary)
    eos = model1.dictionary.eos()
    bos = model1.dictionary.bos()
    rev_dict = dict()
    for k in model1.dictionary.indices.keys():
        v = model1.dictionary.indices[k]
        rev_dict[v] = k
    # pres = []
    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f" % (mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count)))
            batch_size = samples["source"].size(0)
            candidates1 = [dict() for i in range(batch_size)]
            candidates2 = [dict() for i in range(batch_size)]
            candidates3 = [dict() for i in range(batch_size)]
            candidates_path1 = [dict() for i in range(batch_size)]
            candidates_path2 = [dict() for i in range(batch_size)]
            candidates_path3 = [dict() for i in range(batch_size)]
            source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            prefix1 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix2 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix3 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix1[:, :, 0].fill_(model1.dictionary.bos())
            prefix2[:, :, 0].fill_(model2.dictionary.bos())
            prefix3[:, :, 0].fill_(model2.dictionary.bos())
            lprob1 = torch.zeros([batch_size, beam_size]).to(device)
            lprob2 = torch.zeros([batch_size, beam_size]).to(device)
            lprob3 = torch.zeros([batch_size, beam_size]).to(device)
            clen1 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            clen2 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            clen3 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token: choose beam_size from only vocab_size, initiate prefix
            tmp_source1 = samples["source"]
            tmp_source2 = samples["source"]
            tmp_source3 = samples["source"]
            tmp_prefix1 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix2 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix3 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix1[:, 0].fill_(model1.dictionary.bos())
            tmp_prefix2[:, 0].fill_(model2.dictionary.bos())
            tmp_prefix3[:, 0].fill_(model2.dictionary.bos())
            logits1 = model1.logits(tmp_source1, tmp_prefix1).squeeze()
            logits2 = model2.logits(tmp_source2, tmp_prefix2).squeeze()
            logits3 = model2.logits(tmp_source3, tmp_prefix3).squeeze()
            logits1 = F.log_softmax(logits1, dim=-1)
            logits2 = F.log_softmax(logits2, dim=-1)
            logits3 = F.log_softmax(logits3, dim=-1)
            logits1 = logits1.view(-1, vocab_size)
            logits2 = logits2.view(-1, vocab_size)
            logits3 = logits3.view(-1, vocab_size)
            argsort1 = torch.argsort(logits1, dim=-1, descending=True)[:, :beam_size]
            argsort2 = torch.argsort(logits2, dim=-1, descending=True)[:, :beam_size]
            argsort3 = torch.argsort(logits3, dim=-1, descending=True)[:, :beam_size]
            prefix1[:, :, 1] = argsort1[:, :]
            prefix2[:, :, 1] = argsort2[:, :]
            prefix3[:, :, 1] = argsort3[:, :]
            lprob1 += torch.gather(input=logits1, dim=-1, index=argsort1)
            lprob2 += torch.gather(input=logits2, dim=-1, index=argsort2)
            lprob3 += torch.gather(input=logits3, dim=-1, index=argsort3)
            clen1 += 1
            clen2 += 1
            clen3 += 1
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix1 = prefix1.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_prefix2 = prefix2.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_prefix3 = prefix3.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob1 = lprob1.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_lprob2 = lprob2.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_lprob3 = lprob3.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen1 = clen1.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen2 = clen2.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen3 = clen3.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                if l // 2 == 0:
                    all_logits3 = model1.logits(source.view(bb, -1), prefix3.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                else:
                    all_logits3 = model2.logits(source.view(bb, -1), prefix3.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                all_logits1 = model1.logits(source.view(bb, -1), prefix1.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                all_logits2 = model2.logits(source.view(bb, -1), prefix2.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits1 = torch.gather(input=all_logits1, dim=2, index=clen1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                logits2 = torch.gather(input=all_logits2, dim=2, index=clen2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                logits3 = torch.gather(input=all_logits3, dim=2, index=clen3.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # restrict to true_triples, compute index for true_triples
                logits1 = F.log_softmax(logits1, dim=-1)
                logits2 = F.log_softmax(logits2, dim=-1)
                logits3 = F.log_softmax(logits3, dim=-1)
                argsort1 = torch.argsort(logits1, dim=-1, descending=True)[:, :, :beam_size]
                argsort2 = torch.argsort(logits2, dim=-1, descending=True)[:, :, :beam_size]
                argsort3 = torch.argsort(logits3, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen1 = tmp_clen1 + 1
                tmp_clen2 = tmp_clen2 + 1
                tmp_clen3 = tmp_clen3 + 1
                tmp_prefix1 = tmp_prefix1.scatter_(dim=-1, index=tmp_clen1.unsqueeze(-1), src=argsort1.unsqueeze(-1))
                tmp_prefix2 = tmp_prefix2.scatter_(dim=-1, index=tmp_clen2.unsqueeze(-1), src=argsort2.unsqueeze(-1))
                tmp_prefix3 = tmp_prefix3.scatter_(dim=-1, index=tmp_clen3.unsqueeze(-1), src=argsort3.unsqueeze(-1))
                tmp_lprob1 += torch.gather(input=logits1, dim=-1, index=argsort1)
                tmp_lprob2 += torch.gather(input=logits2, dim=-1, index=argsort2)
                tmp_lprob3 += torch.gather(input=logits3, dim=-1, index=argsort3)
                tmp_prefix1, tmp_lprob1, tmp_clen1 = tmp_prefix1.view(batch_size, -1, max_len), tmp_lprob1.view(batch_size, -1), tmp_clen1.view(batch_size, -1)
                tmp_prefix2, tmp_lprob2, tmp_clen2 = tmp_prefix2.view(batch_size, -1, max_len), tmp_lprob2.view(batch_size, -1), tmp_clen2.view(batch_size, -1)
                tmp_prefix3, tmp_lprob3, tmp_clen3 = tmp_prefix3.view(batch_size, -1, max_len), tmp_lprob3.view(batch_size, -1), tmp_clen3.view(batch_size, -1)
                if l == max_len-1:
                    argsort1 = torch.argsort(tmp_lprob1, dim=-1, descending=True)[:, :(2 * beam_size)]
                    argsort2 = torch.argsort(tmp_lprob2, dim=-1, descending=True)[:, :(2 * beam_size)]
                    argsort3 = torch.argsort(tmp_lprob3, dim=-1, descending=True)[:, :(2 * beam_size)]
                else:
                    argsort1 = torch.argsort(tmp_lprob1, dim=-1, descending=True)[:, :beam_size]
                    argsort2 = torch.argsort(tmp_lprob2, dim=-1, descending=True)[:, :beam_size]
                    argsort3 = torch.argsort(tmp_lprob3, dim=-1, descending=True)[:, :beam_size]
                prefix1 = torch.gather(input=tmp_prefix1, dim=1, index=argsort1.unsqueeze(-1).repeat(1, 1, max_len))
                prefix2 = torch.gather(input=tmp_prefix2, dim=1, index=argsort2.unsqueeze(-1).repeat(1, 1, max_len))
                prefix3 = torch.gather(input=tmp_prefix3, dim=1, index=argsort3.unsqueeze(-1).repeat(1, 1, max_len))
                lprob1 = torch.gather(input=tmp_lprob1, dim=1, index=argsort1)
                lprob2 = torch.gather(input=tmp_lprob2, dim=1, index=argsort2)
                lprob3 = torch.gather(input=tmp_lprob3, dim=1, index=argsort3)
                clen1 = torch.gather(input=tmp_clen1, dim=1, index=argsort1)
                clen2 = torch.gather(input=tmp_clen2, dim=1, index=argsort2)
                clen3 = torch.gather(input=tmp_clen3, dim=1, index=argsort3)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if prefix1[i][j][l].item() == eos:
                            candidate1 = prefix1[i][j][l-1].item()
                            if l_punish:
                                prob1 = lprob1[i][j].item() / int(l / 2)
                            else:
                                prob1 = lprob1[i][j].item()
                            lprob1[i][j] -= 10000
                            if candidate1 not in candidates1[i]:
                                candidates1[i][candidate1] = prob1
                                candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                            else:
                                if prob1 > candidates1[i][candidate1]:
                                    candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                                candidates1[i][candidate1] = max(candidates1[i][candidate1], prob1)
                        if prefix2[i][j][l].item() == eos:
                            candidate2 = prefix2[i][j][l-1].item()
                            if l_punish:
                                prob2 = lprob2[i][j].item() / int(l / 2)
                            else:
                                prob2 = lprob2[i][j].item()
                            lprob2[i][j] -= 10000
                            if candidate2 not in candidates2[i]:
                                candidates2[i][candidate2] = prob2
                                candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                            else:
                                if prob2 > candidates2[i][candidate2]:
                                    candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                                candidates2[i][candidate2] = max(candidates2[i][candidate2], prob2)
                        if prefix3[i][j][l].item() == eos:
                            candidate3 = prefix3[i][j][l-1].item()
                            if l_punish:
                                prob3 = lprob3[i][j].item() / int(l / 2)
                            else:
                                prob3 = lprob3[i][j].item()
                            lprob3[i][j] -= 10000
                            if candidate3 not in candidates3[i]:
                                candidates3[i][candidate3] = prob3
                                candidates_path3[i][candidate3] = prefix3[i][j].cpu().numpy()
                            else:
                                if prob3 > candidates3[i][candidate3]:
                                    candidates_path3[i][candidate3] = prefix3[i][j].cpu().numpy()
                                candidates3[i][candidate3] = max(candidates3[i][candidate3], prob3)
                # no <end> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate1 = prefix1[i][j][l].item()
                            candidate2 = prefix2[i][j][l].item()
                            candidate3 = prefix3[i][j][l].item()
                            if l_punish:
                                prob1 = lprob1[i][j].item() / int(max_len / 2)
                                prob2 = lprob2[i][j].item() / int(max_len / 2)
                                prob3 = lprob3[i][j].item() / int(max_len / 2)
                            else:
                                prob1 = lprob1[i][j].item()
                                prob2 = lprob2[i][j].item()
                                prob3 = lprob3[i][j].item()
                            if candidate1 not in candidates1[i]:
                                candidates1[i][candidate1] = prob1
                                candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                            else:
                                if prob1 > candidates1[i][candidate1]:
                                    candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                                candidates1[i][candidate1] = max(candidates1[i][candidate1], prob1)
                            if candidate2 not in candidates2[i]:
                                candidates2[i][candidate2] = prob2
                                candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                            else:
                                if prob2 > candidates2[i][candidate2]:
                                    candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                                candidates2[i][candidate2] = max(candidates2[i][candidate2], prob2)
                            if candidate3 not in candidates3[i]:
                                candidates3[i][candidate3] = prob3
                                candidates_path3[i][candidate3] = prefix3[i][j].cpu().numpy()
                            else:
                                if prob3 > candidates3[i][candidate3]:
                                    candidates_path3[i][candidate3] = prefix3[i][j].cpu().numpy()
                                candidates3[i][candidate3] = max(candidates3[i][candidate3], prob3)
            target = samples["target"].cpu()
            for i in range(batch_size):
                hid = samples["source"][i][-2].item()
                rid = samples["source"][i][-1].item()
                index = vocab_size * rid + hid
                if index in valid_triples:
                    mask = valid_triples[index]
                    for tid in candidates1[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates1[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates1[i][tid] -= 100000
                    for tid in candidates2[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates2[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates2[i][tid] -= 100000
                    for tid in candidates3[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates3[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates3[i][tid] -= 100000
                count += 1
                candidate_1 = sorted(zip(candidates1[i].items(), candidates_path1[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate_2 = sorted(zip(candidates2[i].items(), candidates_path2[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate_3 = sorted(zip(candidates3[i].items(), candidates_path3[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate1 = [pair[0][0] for pair in candidate_1]
                candidate2 = [pair[0][0] for pair in candidate_2]
                candidate3 = [pair[0][0] for pair in candidate_3]
                candidate_entity_prob1 = [pair[0] for pair in candidate_1]
                candidate_entity_prob2 = [pair[0] for pair in candidate_2]
                candidate_entity_prob3 = [pair[0] for pair in candidate_3]
                candidate_dict = {}
                for entity,prob in candidate_entity_prob1:
                    if entity not in candidate_dict.keys():
                        candidate_dict[entity] = prob
                    else:
                        candidate_dict[entity] += prob
                for entity,prob in candidate_entity_prob2:
                    if entity not in candidate_dict.keys():
                        candidate_dict[entity] = prob
                    else:
                        candidate_dict[entity] += prob
                for entity,prob in candidate_entity_prob3:
                    if entity not in candidate_dict.keys():
                        candidate_dict[entity] = prob
                    else:
                        candidate_dict[entity] += prob
                candidate = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)
                candidate = [pair[0] for pair in candidate]
                candidate_fisrt = []
                candidate_second = []
                candidate_third = []
                for candi in candidate:
                    if candi in candidate1 and candi in candidate2 and candi in candidate3:
                        candidate_fisrt.append(candi)
                    elif (candi in candidate1 and candi in candidate2) or (candi in candidate1 and candi in candidate3) or (candi in candidate2 and candi in candidate3):
                        candidate_second.append(candi)
                    else:
                        candidate_third.append(candi)
                candidate_final = []
                candidate_final.extend(candidate_fisrt)
                candidate_final.extend(candidate_second)
                candidate_final.extend(candidate_third)
                candidate_final = torch.tensor(candidate_final)
                ranking = (candidate_final[:] == target[i]).nonzero()
                if ranking.nelement() != 0:
                    ranking = 1 + ranking.item()
                    mrr += (1 / ranking)
                    hit += 1
                    if ranking <= 1:
                        hit1 += 1
                    if ranking <= 3:
                        hit3 += 1
                    if ranking <= 10:
                        hit10 += 1
    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]" % (mrr/count, hit1/count, hit3/count, hit10/count))
    return hit/count, hit1/count, hit3/count, hit10/count

def evaluate2(model1, model2, dataloader, device, args, true_triples=None, valid_triples=None, epoch=None):
    model1.eval()
    model2.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
    vocab_size = len(model1.dictionary)
    eos = model1.dictionary.eos()
    bos = model1.dictionary.bos()
    rev_dict = dict()
    for k in model1.dictionary.indices.keys():
        v = model1.dictionary.indices[k]
        rev_dict[v] = k
    lines = []
    # pres = []
    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f" % (mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count)))
            batch_size = samples["source"].size(0)
            candidates1 = [dict() for i in range(batch_size)]
            candidates2 = [dict() for i in range(batch_size)]
            candidates_path1 = [dict() for i in range(batch_size)]
            candidates_path2 = [dict() for i in range(batch_size)]
            source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            prefix1 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix2 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix1[:, :, 0].fill_(model1.dictionary.bos())
            prefix2[:, :, 0].fill_(model2.dictionary.bos())
            lprob1 = torch.zeros([batch_size, beam_size]).to(device)
            lprob2 = torch.zeros([batch_size, beam_size]).to(device)
            clen1 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            clen2 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token: choose beam_size from only vocab_size, initiate prefix
            tmp_source1 = samples["source"]
            tmp_source2 = samples["source"]
            tmp_prefix1 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix2 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix1[:, 0].fill_(model1.dictionary.bos())
            tmp_prefix2[:, 0].fill_(model2.dictionary.bos())
            logits1 = model1.logits(tmp_source1, tmp_prefix1).squeeze()
            logits2 = model2.logits(tmp_source2, tmp_prefix2).squeeze()
            logits1 = F.log_softmax(logits1, dim=-1)
            logits2 = F.log_softmax(logits2, dim=-1)
            logits1 = logits1.view(-1, vocab_size)
            logits2 = logits2.view(-1, vocab_size)
            argsort1 = torch.argsort(logits1, dim=-1, descending=True)[:, :beam_size]
            argsort2 = torch.argsort(logits2, dim=-1, descending=True)[:, :beam_size]
            prefix1[:, :, 1] = argsort1[:, :]
            prefix2[:, :, 1] = argsort2[:, :]
            lprob1 += torch.gather(input=logits1, dim=-1, index=argsort1)
            lprob2 += torch.gather(input=logits2, dim=-1, index=argsort2)
            clen1 += 1
            clen2 += 1
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix1 = prefix1.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_prefix2 = prefix2.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob1 = lprob1.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_lprob2 = lprob2.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen1 = clen1.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen2 = clen2.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                all_logits1 = model1.logits(source.view(bb, -1), prefix1.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                all_logits2 = model2.logits(source.view(bb, -1), prefix2.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits1 = torch.gather(input=all_logits1, dim=2, index=clen1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                logits2 = torch.gather(input=all_logits2, dim=2, index=clen2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # restrict to true_triples, compute index for true_triples
                logits1 = F.log_softmax(logits1, dim=-1)
                logits2 = F.log_softmax(logits2, dim=-1)
                argsort1 = torch.argsort(logits1, dim=-1, descending=True)[:, :, :beam_size]
                argsort2 = torch.argsort(logits2, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen1 = tmp_clen1 + 1
                tmp_clen2 = tmp_clen2 + 1
                tmp_prefix1 = tmp_prefix1.scatter_(dim=-1, index=tmp_clen1.unsqueeze(-1), src=argsort1.unsqueeze(-1))
                tmp_prefix2 = tmp_prefix2.scatter_(dim=-1, index=tmp_clen2.unsqueeze(-1), src=argsort2.unsqueeze(-1))
                tmp_lprob1 += torch.gather(input=logits1, dim=-1, index=argsort1)
                tmp_lprob2 += torch.gather(input=logits2, dim=-1, index=argsort2)
                tmp_prefix1, tmp_lprob1, tmp_clen1 = tmp_prefix1.view(batch_size, -1, max_len), tmp_lprob1.view(batch_size, -1), tmp_clen1.view(batch_size, -1)
                tmp_prefix2, tmp_lprob2, tmp_clen2 = tmp_prefix2.view(batch_size, -1, max_len), tmp_lprob2.view(batch_size, -1), tmp_clen2.view(batch_size, -1)
                if l == max_len-1:
                    argsort1 = torch.argsort(tmp_lprob1, dim=-1, descending=True)[:, :(2 * beam_size)]
                    argsort2 = torch.argsort(tmp_lprob2, dim=-1, descending=True)[:, :(2 * beam_size)]
                else:
                    argsort1 = torch.argsort(tmp_lprob1, dim=-1, descending=True)[:, :beam_size]
                    argsort2 = torch.argsort(tmp_lprob2, dim=-1, descending=True)[:, :beam_size]
                prefix1 = torch.gather(input=tmp_prefix1, dim=1, index=argsort1.unsqueeze(-1).repeat(1, 1, max_len))
                prefix2 = torch.gather(input=tmp_prefix2, dim=1, index=argsort2.unsqueeze(-1).repeat(1, 1, max_len))
                lprob1 = torch.gather(input=tmp_lprob1, dim=1, index=argsort1)
                lprob2 = torch.gather(input=tmp_lprob2, dim=1, index=argsort2)
                clen1 = torch.gather(input=tmp_clen1, dim=1, index=argsort1)
                clen2 = torch.gather(input=tmp_clen2, dim=1, index=argsort2)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if prefix1[i][j][l].item() == eos:
                            candidate1 = prefix1[i][j][l-1].item()
                            if l_punish:
                                prob1 = lprob1[i][j].item() / int(l / 2)
                            else:
                                prob1 = lprob1[i][j].item()
                            lprob1[i][j] -= 10000
                            if candidate1 not in candidates1[i]:
                                if args.self_consistency:
                                    candidates1[i][candidate1] = math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = prob1
                                candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                            else:
                                if prob1 > candidates1[i][candidate1]:
                                    candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates1[i][candidate1] += math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = max(candidates1[i][candidate1], prob1)
                        if prefix2[i][j][l].item() == eos:
                            candidate2 = prefix2[i][j][l-1].item()
                            if l_punish:
                                prob2 = lprob2[i][j].item() / int(l / 2)
                            else:
                                prob2 = lprob2[i][j].item()
                            lprob2[i][j] -= 10000
                            if candidate2 not in candidates2[i]:
                                if args.self_consistency:
                                    candidates2[i][candidate2] = math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = prob2
                                candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                            else:
                                if prob2 > candidates2[i][candidate2]:
                                    candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates2[i][candidate2] += math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = max(candidates2[i][candidate2], prob2)
                # no <end> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate1 = prefix1[i][j][l].item()
                            candidate2 = prefix2[i][j][l].item()
                            if l_punish:
                                prob1 = lprob1[i][j].item() / int(max_len / 2)
                                prob2 = lprob2[i][j].item() / int(max_len / 2)
                            else:
                                prob1 = lprob1[i][j].item()
                                prob2 = lprob2[i][j].item()
                            if candidate1 not in candidates1[i]:
                                if args.self_consistency:
                                    candidates1[i][candidate1] = math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = prob1
                                candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                            else:
                                if prob1 > candidates1[i][candidate1]:
                                    candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates1[i][candidate1] += math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = max(candidates1[i][candidate1], prob1)
                            if candidate2 not in candidates2[i]:
                                if args.self_consistency:
                                    candidates2[i][candidate2] = math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = prob2
                                candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                            else:
                                if prob2 > candidates2[i][candidate2]:
                                    candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates2[i][candidate2] += math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = max(candidates2[i][candidate2], prob2)

            target = samples["target"].cpu()
            for i in range(batch_size):
                hid = samples["source"][i][-2].item()
                rid = samples["source"][i][-1].item()
                index = vocab_size * rid + hid
                if index in valid_triples:
                    mask = valid_triples[index]
                    for tid in candidates1[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates1[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates1[i][tid] -= 100000
                    for tid in candidates2[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates2[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates2[i][tid] -= 100000
                count += 1
                candidate_1 = sorted(zip(candidates1[i].items(), candidates_path1[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate_2 = sorted(zip(candidates2[i].items(), candidates_path2[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate1 = [pair[0][0] for pair in candidate_1]
                candidate2 = [pair[0][0] for pair in candidate_2]
                candidate_entity_prob1 = [pair[0] for pair in candidate_1]
                candidate_entity_prob2 = [pair[0] for pair in candidate_2]

                if args.output_path:
                    path_token = rev_dict[hid] + " " + rev_dict[rid] + " " + rev_dict[target[i].item()] + '\t'
                    candidate_path1 = [pair[1][1] for pair in candidate_1]
                    candidate_path2 = [pair[1][1] for pair in candidate_2]
                    candidate1 = torch.from_numpy(np.array(candidate1))
                    candidate2 = torch.from_numpy(np.array(candidate2))
                    ranking1 = (candidate1[:] == target[i]).nonzero()
                    ranking2 = (candidate2[:] == target[i]).nonzero()
                    dict_entity_prob1 = {}
                    for entity, prob in candidate_entity_prob1:
                        dict_entity_prob1[entity] = prob
                    dict_entity_prob2 = {}
                    for entity, prob in candidate_entity_prob2:
                        dict_entity_prob2[entity] = prob
                    if ranking1.nelement() != 0 and ranking2.nelement() != 0:
                        #if ranking1 >= ranking2:
                        if dict_entity_prob2[target[i].item()] >= dict_entity_prob1[target[i].item()]:
                            path = candidate_path2[ranking2]
                        else:
                            path = candidate_path1[ranking1]
                        for token in path[1:-1]:
                            path_token += (rev_dict[token] + ' ')
                        path_token += (rev_dict[path[-1]] + '\t')
                    elif ranking1.nelement() != 0 and ranking2.nelement() == 0:
                        path = candidate_path1[ranking1]
                        for token in path[1:-1]:
                            path_token += (rev_dict[token] + ' ')
                        path_token += (rev_dict[path[-1]] + '\t')
                    elif ranking1.nelement() == 0 and ranking2.nelement() != 0:
                        path = candidate_path2[ranking2]
                        for token in path[1:-1]:
                            path_token += (rev_dict[token] + ' ')
                        path_token += (rev_dict[path[-1]] + '\t')
                    else:
                        path_token += "wrong"


                candidate_dict = {}
                for entity,prob in candidate_entity_prob1:
                    if entity not in candidate_dict.keys():
                        candidate_dict[entity] = prob
                    else:
                        candidate_dict[entity] += prob
                for entity,prob in candidate_entity_prob2:
                    if entity not in candidate_dict.keys():
                        candidate_dict[entity] = prob
                    else:
                        candidate_dict[entity] += prob
                candidate = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)
                candidate = [pair[0] for pair in candidate]
                candidate_fisrt = []
                candidate_second = []
                for candi in candidate:
                    if candi in candidate1 and candi in candidate2:
                        candidate_fisrt.append(candi)
                    else:
                        candidate_second.append(candi)
                candidate_final = []
                candidate_final.extend(candidate_fisrt)
                candidate_final.extend(candidate_second)
                candidate_final = torch.tensor(candidate_final)
                ranking = (candidate_final[:] == target[i]).nonzero()
                if args.output_path:
                    if ranking.nelement() != 0:
                        path_token += str(ranking.item())
                    lines.append(path_token + '\n')
                if ranking.nelement() != 0:
                    ranking = 1 + ranking.item()
                    mrr += (1 / ranking)
                    hit += 1
                    if ranking <= 1:
                        hit1 += 1
                    if ranking <= 3:
                        hit3 += 1
                    if ranking <= 10:
                        hit10 += 1
    if args.output_path:
        with open("paths/selfhier_" + str(args.dataset).strip('data/') + "_" + str(args.seed) + "_epoch" + "_" + str(epoch) + ".txt", "w") as f:
            f.writelines(lines)
    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]" % (mrr/count, hit1/count, hit3/count, hit10/count))
    return hit/count, hit1/count, hit3/count, hit10/count

def evaluate3(model1, model2, dataloader, device, args, true_triples=None, valid_triples=None):
    all_triplets = {}
    trains = []
    with open(args.dataset + '/train_triples_rev.txt', 'r') as fin:
        for line in fin:
            trains.append(line.strip())
    model1.eval()
    model2.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
    vocab_size = len(model1.dictionary)
    eos = model1.dictionary.eos()
    bos = model1.dictionary.bos()
    rev_dict = dict()
    for k in model1.dictionary.indices.keys():
        v = model1.dictionary.indices[k]
        rev_dict[v] = k
    # pres = []
    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("count: %f, " % count)
            batch_size = samples["source"].size(0)
            candidates1 = [dict() for i in range(batch_size)]
            candidates2 = [dict() for i in range(batch_size)]
            candidates_path1 = [dict() for i in range(batch_size)]
            candidates_path2 = [dict() for i in range(batch_size)]
            source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            prefix1 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix2 = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix1[:, :, 0].fill_(model1.dictionary.bos())
            prefix2[:, :, 0].fill_(model2.dictionary.bos())
            lprob1 = torch.zeros([batch_size, beam_size]).to(device)
            lprob2 = torch.zeros([batch_size, beam_size]).to(device)
            clen1 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            clen2 = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token: choose beam_size from only vocab_size, initiate prefix
            tmp_source1 = samples["source"]
            tmp_source2 = samples["source"]
            tmp_prefix1 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix2 = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix1[:, 0].fill_(model1.dictionary.bos())
            tmp_prefix2[:, 0].fill_(model2.dictionary.bos())
            logits1 = model1.logits(tmp_source1, tmp_prefix1).squeeze()
            logits2 = model2.logits(tmp_source2, tmp_prefix2).squeeze()
            logits1 = F.log_softmax(logits1, dim=-1)
            logits2 = F.log_softmax(logits2, dim=-1)
            logits1 = logits1.view(-1, vocab_size)
            logits2 = logits2.view(-1, vocab_size)
            argsort1 = torch.argsort(logits1, dim=-1, descending=True)[:, :beam_size]
            argsort2 = torch.argsort(logits2, dim=-1, descending=True)[:, :beam_size]
            prefix1[:, :, 1] = argsort1[:, :]
            prefix2[:, :, 1] = argsort2[:, :]
            lprob1 += torch.gather(input=logits1, dim=-1, index=argsort1)
            lprob2 += torch.gather(input=logits2, dim=-1, index=argsort2)
            clen1 += 1
            clen2 += 1
            target = samples["target"].cpu()
            for l in range(2, max_len):
                tmp_prefix1 = prefix1.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_prefix2 = prefix2.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob1 = lprob1.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_lprob2 = lprob2.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen1 = clen1.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                tmp_clen2 = clen2.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                all_logits1 = model1.logits(source.view(bb, -1), prefix1.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                all_logits2 = model2.logits(source.view(bb, -1), prefix2.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits1 = torch.gather(input=all_logits1, dim=2, index=clen1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                logits2 = torch.gather(input=all_logits2, dim=2, index=clen2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # restrict to true_triples, compute index for true_triples
                logits1 = F.log_softmax(logits1, dim=-1)
                logits2 = F.log_softmax(logits2, dim=-1)
                argsort1 = torch.argsort(logits1, dim=-1, descending=True)[:, :, :beam_size]
                argsort2 = torch.argsort(logits2, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen1 = tmp_clen1 + 1
                tmp_clen2 = tmp_clen2 + 1
                tmp_prefix1 = tmp_prefix1.scatter_(dim=-1, index=tmp_clen1.unsqueeze(-1), src=argsort1.unsqueeze(-1))
                tmp_prefix2 = tmp_prefix2.scatter_(dim=-1, index=tmp_clen2.unsqueeze(-1), src=argsort2.unsqueeze(-1))
                tmp_lprob1 += torch.gather(input=logits1, dim=-1, index=argsort1)
                tmp_lprob2 += torch.gather(input=logits2, dim=-1, index=argsort2)
                tmp_prefix1, tmp_lprob1, tmp_clen1 = tmp_prefix1.view(batch_size, -1, max_len), tmp_lprob1.view(batch_size, -1), tmp_clen1.view(batch_size, -1)
                tmp_prefix2, tmp_lprob2, tmp_clen2 = tmp_prefix2.view(batch_size, -1, max_len), tmp_lprob2.view(batch_size, -1), tmp_clen2.view(batch_size, -1)
                if l == max_len-1:
                    argsort1 = torch.argsort(tmp_lprob1, dim=-1, descending=True)[:, :(2 * beam_size)]
                    argsort2 = torch.argsort(tmp_lprob2, dim=-1, descending=True)[:, :(2 * beam_size)]
                else:
                    argsort1 = torch.argsort(tmp_lprob1, dim=-1, descending=True)[:, :beam_size]
                    argsort2 = torch.argsort(tmp_lprob2, dim=-1, descending=True)[:, :beam_size]
                prefix1 = torch.gather(input=tmp_prefix1, dim=1, index=argsort1.unsqueeze(-1).repeat(1, 1, max_len))
                prefix2 = torch.gather(input=tmp_prefix2, dim=1, index=argsort2.unsqueeze(-1).repeat(1, 1, max_len))
                lprob1 = torch.gather(input=tmp_lprob1, dim=1, index=argsort1)
                lprob2 = torch.gather(input=tmp_lprob2, dim=1, index=argsort2)
                clen1 = torch.gather(input=tmp_clen1, dim=1, index=argsort1)
                clen2 = torch.gather(input=tmp_clen2, dim=1, index=argsort2)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if prefix1[i][j][l].item() == eos:
                            candidate1 = prefix1[i][j][l-1].item()
                            if l_punish:
                                prob1 = lprob1[i][j].item() / int(l / 2)
                            else:
                                prob1 = lprob1[i][j].item()
                            lprob1[i][j] -= 10000
                            if candidate1 not in candidates1[i]:
                                if args.self_consistency:
                                    candidates1[i][candidate1] = math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = prob1
                                candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                            else:
                                if prob1 > candidates1[i][candidate1]:
                                    candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates1[i][candidate1] += math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = max(candidates1[i][candidate1], prob1)
                        if prefix2[i][j][l].item() == eos:
                            candidate2 = prefix2[i][j][l-1].item()
                            if l_punish:
                                prob2 = lprob2[i][j].item() / int(l / 2)
                            else:
                                prob2 = lprob2[i][j].item()
                            lprob2[i][j] -= 10000
                            if candidate2 not in candidates2[i]:
                                if args.self_consistency:
                                    candidates2[i][candidate2] = math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = prob2
                                candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                            else:
                                if prob2 > candidates2[i][candidate2]:
                                    candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates2[i][candidate2] += math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = max(candidates2[i][candidate2], prob2)
                # no <end> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate1 = prefix1[i][j][l].item()
                            candidate2 = prefix2[i][j][l].item()
                            if l_punish:
                                prob1 = lprob1[i][j].item() / int(max_len / 2)
                                prob2 = lprob2[i][j].item() / int(max_len / 2)
                            else:
                                prob1 = lprob1[i][j].item()
                                prob2 = lprob2[i][j].item()
                            if candidate1 not in candidates1[i]:
                                if args.self_consistency:
                                    candidates1[i][candidate1] = math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = prob1
                                candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                            else:
                                if prob1 > candidates1[i][candidate1]:
                                    candidates_path1[i][candidate1] = prefix1[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates1[i][candidate1] += math.exp(prob1)
                                else:
                                    candidates1[i][candidate1] = max(candidates1[i][candidate1], prob1)
                            if candidate2 not in candidates2[i]:
                                if args.self_consistency:
                                    candidates2[i][candidate2] = math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = prob2
                                candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                            else:
                                if prob2 > candidates2[i][candidate2]:
                                    candidates_path2[i][candidate2] = prefix2[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates2[i][candidate2] += math.exp(prob2)
                                else:
                                    candidates2[i][candidate2] = max(candidates2[i][candidate2], prob2)

            target = samples["target"].cpu()
            for i in range(batch_size):
                hid = samples["source"][i][-2].item()
                rid = samples["source"][i][-1].item()
                index = vocab_size * rid + hid
                if index in valid_triples:
                    mask = valid_triples[index]
                    for tid in candidates1[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates1[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates1[i][tid] -= 100000
                    for tid in candidates2[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates2[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates2[i][tid] -= 100000
                count += 1
                candidate_1 = sorted(zip(candidates1[i].items(), candidates_path1[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate_2 = sorted(zip(candidates2[i].items(), candidates_path2[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate1 = [pair[0][0] for pair in candidate_1]
                candidate2 = [pair[0][0] for pair in candidate_2]
                candidate1 = torch.from_numpy(np.array(candidate1))
                candidate2 = torch.from_numpy(np.array(candidate2))
                candidate_path1 = [pair[1][1] for pair in candidate_1]
                candidate_path2 = [pair[1][1] for pair in candidate_2]
                ranking1 = (candidate1[:] == target[i]).nonzero()
                ranking2 = (candidate2[:] == target[i]).nonzero()
                if ranking1.nelement() != 0:
                    path1 = candidate_path1[ranking1]
                    path1[0] = hid
                if ranking2.nelement() != 0:
                    path2 = candidate_path2[ranking2]
                    path2[0] = hid
                path_token1 = []
                for token in path1:
                    if token != 0 and token != 2:
                        path_token1.append(rev_dict[token])
                path_token2 = []
                for token in path2:
                    if token != 0 and token != 2:
                        path_token2.append(rev_dict[token])
                for j in range(int((len(path_token1)-1)/2)):
                    triplet = str(path_token1[j*2]) + '\t' + str(path_token1[j*2+1]) + '\t' + str(path_token1[j*2+2])
                    if triplet not in trains:
                        if triplet not in all_triplets.keys():
                            all_triplets[triplet] = 1
                        else:
                            all_triplets[triplet] += 1
                for j in range(int((len(path_token2)-1)/2)):
                    triplet = str(path_token2[j*2]) + '\t' + str(path_token2[j*2+1]) + '\t' + str(path_token2[j*2+2])
                    if triplet not in trains:
                        if triplet not in all_triplets.keys():
                            all_triplets[triplet] = 1
                        else:
                            all_triplets[triplet] += 1
    with open(args.dataset + '/new_dict.json', 'w') as f:
        f.write(json.dumps(all_triplets))
    logging.info("[count: %f]" % (count))
    return

def get_cluster_relation_cross_loss(cluster_prob, relation_prob, Clu2Ent_matrix,Rel2rel_matrix, device):
    relation2entity = torch.spmm(Rel2rel_matrix,relation_prob.view(-1,relation_prob.size()[-1]).transpose(0,1))
    relation2cluster = torch.spmm(Clu2Ent_matrix, relation2entity).transpose(0,1).view(cluster_prob.size()[0],cluster_prob.size()[1],-1)
    relation2cluster = relation2cluster.view(-1, relation2cluster.size()[-1])
    cluster_prob = cluster_prob.view(-1, cluster_prob.size()[-1])
    target = torch.ones(relation2cluster.size()[0]).to(device)
    criterion = nn.CosineEmbeddingLoss(reduction='mean')
    return criterion(relation2cluster,cluster_prob,target)

def train(args):
    print('alpha:{0}, gmma:{1}, beta:{2}, theta:{3}, epsilon:{4}, loss:{5}, seed:{6}, K:{7}, Dataset:{8}, lr:{9}, label_smooth:{10}, dropout{11}, num_layers:{12}'.
          format(args.alpha, args.gmma, args.beta, args.theta, args.epsilon, args.loss, args.seed, args.K, args.dataset, args.lr, args.label_smooth,args.dropout, args.num_layers))
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models_new', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/train.log',
                    filemode='w',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.info(args)
    device = "cuda:{0}".format(args.cuda) if torch.cuda.is_available() else "cpu"
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", device=device, args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    train_valid, eval_valid = train_set.get_next_valid()
    test_set = TestDataset(data_path=args.dataset + "/", device=device, src_file="test_triples.txt", args=args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    pre_test_set = TestDataset(data_path=args.dataset + "/", device=device, src_file="train_triples_rev.txt", args=args)
    pre_test_loader = DataLoader(pre_test_set, batch_size=args.test_batch_size, collate_fn=pre_test_set.collate_fn,shuffle=True)
    if args.valid:
        valid_set = TestDataset(data_path=args.dataset + "/", device=device,src_file="valid_triples.txt", args=args)
        valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    if args.cluster:
        model_entity_cluster = TransformerModel(args, train_set.entity_dictionary, 'entity').to(device)
        model_cluster = TransformerModel(args, train_set.cluster_dictionary, 'cluster').to(device)
    if args.relation:
        model_entity_relation = TransformerModel(args, train_set.entity_dictionary, 'entity').to(device)
        model_relation = TransformerModel(args, train_set.relation_dictionary, 'relation').to(device)
    if not args.cluster and not args.relation:
        model_entity = TransformerModel(args, train_set.entity_dictionary, 'entity').to(device)
    if args.cluster and args.relation:
        optimizer = optim.Adam(list(model_entity_cluster.parameters())+
                               list(model_entity_relation.parameters())+
                               list(model_cluster.parameters())+
                               list(model_relation.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)
    elif args.cluster:
        optimizer = optim.Adam(
            list(model_entity_cluster.parameters()) +
            list(model_cluster.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
    elif args.relation:
        optimizer = optim.Adam(
            list(model_entity_relation.parameters()) +
            list(model_relation.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(
            list(model_entity.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
    total_step_num = len(train_loader) * args.num_epoch
    warmup_steps = total_step_num / args.warmup
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_step_num)
    if args.cluster:
        model_entity_cluster.get_entity_matrix(len(train_set.entity_dictionary.symbols), len(train_set.cluster_dictionary.symbols))
    if args.relation:
        with open(args.dataset + '/entity2id.txt') as fin:
            for line in fin:
                entity_num = int(line.strip())
                print(entity_num)
                break
        model_relation.get_relation_matrix(len(train_set.entity_dictionary.symbols), len(train_set.relation_dictionary.symbols), entity_num)
    steps = 0
    for epoch in range(args.num_epoch):
        if args.cluster and args.relation:
            model_entity_cluster.train()
            model_entity_relation.train()
            model_cluster.train()
            model_relation.train()
            with tqdm(train_loader, desc="training") as pbar:
                losses = []
                e_c_losses = []
                e_r_losses = []
                c_losses = []
                r_losses = []
                cross_cluster_losses = []
                cross_relaiton_losses = []
                cluster_relation_cross_losses = []
                for samples in pbar:
                    ids, lengths, \
                    source_entity, prev_outputs_entity, target_entity, mask_entity, \
                    source_cluster, prev_outputs_cluster, target_cluster, mask_cluster, \
                    source_relation, prev_outputs_relation, target_relation, mask_relation \
                        = samples['ids'], samples['lengths'], \
                          samples['source_entity'], samples['prev_outputs_entity'], samples['target_entity'], samples['mask_entity'],\
                          samples['source_cluster'], samples['prev_outputs_cluster'], samples['target_cluster'], samples['mask_cluster'],\
                          samples['source_relation'], samples['prev_outputs_relation'], samples['target_relation'], samples['mask_relation'],
                    optimizer.zero_grad()
                    loss_entity_cluster, entity_cluster_prob = model_entity_cluster.get_loss(source_entity, prev_outputs_entity, target_entity, mask_entity)
                    loss_entity_relation, entity_relation_prob = model_entity_relation.get_loss(source_entity,prev_outputs_entity, target_entity, mask_entity)
                    loss_cluster, cluster_prob = model_cluster.get_loss(source_cluster, prev_outputs_cluster, target_cluster, mask_cluster)
                    loss_relation, relation_prob = model_relation.get_loss(source_relation, prev_outputs_relation, target_entity, mask_relation)
                    if args.loss == 1:
                        cross_cluster_loss, Clu2Ent_matrix = model_entity_cluster.get_cross_loss1(cluster_prob)
                        cross_relaiton_loss, Rel2rel_matrix = model_relation.get_cross_loss1(entity_relation_prob)
                    elif args.loss == 2:
                        cross_cluster_loss, Clu2Ent_matrix = model_entity_cluster.get_cross_loss2(cluster_prob)
                        cross_relaiton_loss, Rel2rel_matrix = model_relation.get_cross_loss2(entity_relation_prob)
                    cluster_relation_cross_loss = get_cluster_relation_cross_loss(cluster_prob, relation_prob, Clu2Ent_matrix, Rel2rel_matrix, device)
                    loss =  loss_entity_cluster + \
                            loss_entity_relation + \
                            args.epsilon * cluster_relation_cross_loss +\
                            args.alpha * loss_cluster + \
                            args.beta * loss_relation + \
                            args.gmma * cross_cluster_loss + \
                            args.theta * cross_relaiton_loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    losses.append(loss.item())
                    e_c_losses.append(loss_entity_cluster.item())
                    e_r_losses.append(loss_entity_relation.item())
                    c_losses.append(loss_cluster.item())
                    r_losses.append(loss_relation.item())
                    cross_cluster_losses.append(cross_cluster_loss.item())
                    cross_relaiton_losses.append(cross_relaiton_loss.item())
                    cluster_relation_cross_losses.append(cluster_relation_cross_loss.item())
                    pbar.set_description("Epoch: %d, Loss: %0.8f, E_C_loss: %0.8f, E_R_loss: %0.8f, C_loss: %0.8f, R_loss: %0.8f, cross_clu_loss: %0.8f,cross_rel_loss: %0.8f, cross_rel_clu_loss: %0.8f, lr: %0.6f" % (
                    epoch + 1, np.mean(losses), np.mean(e_c_losses), np.mean(e_r_losses), np.mean(c_losses), np.mean(r_losses),np.mean(cross_cluster_losses),np.mean(cross_relaiton_losses), np.mean(cluster_relation_cross_losses),optimizer.param_groups[0]['lr']))
            logging.info(
                "[Epoch %d/%d] [train loss: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses))
            )
        elif args.cluster:
            model_entity_cluster.train()
            model_cluster.train()
            with tqdm(train_loader, desc="training") as pbar:
                losses = []
                e_c_losses = []
                c_losses = []
                cross_cluster_losses = []
                for samples in pbar:
                    ids, lengths, \
                    source_entity, prev_outputs_entity, target_entity, mask_entity, \
                    source_cluster, prev_outputs_cluster, target_cluster, mask_cluster, \
                    source_relation, prev_outputs_relation, target_relation, mask_relation \
                        = samples['ids'], samples['lengths'], \
                          samples['source_entity'], samples['prev_outputs_entity'], samples['target_entity'], samples['mask_entity'],\
                          samples['source_cluster'], samples['prev_outputs_cluster'], samples['target_cluster'], samples['mask_cluster'],\
                          samples['source_relation'], samples['prev_outputs_relation'], samples['target_relation'], samples['mask_relation'],
                    optimizer.zero_grad()
                    loss_entity, entity_prob = model_entity_cluster.get_loss(source_entity, prev_outputs_entity, target_entity, mask_entity)
                    #train_set_entity.dictionary.symbols
                    loss_cluster, cluster_prob = model_cluster.get_loss(source_cluster, prev_outputs_cluster, target_cluster, mask_cluster)
                    if args.loss == 1:
                        cross_cluster_loss,_ = model_entity_cluster.get_cross_loss1(cluster_prob)
                    elif args.loss == 2:
                        cross_cluster_loss,_ = model_entity_cluster.get_cross_loss2(cluster_prob)
                    elif args.loss == 3:
                        cross_cluster_loss, _ = model_entity_cluster.get_cross_loss3(lengths, cluster_prob)
                    loss =  loss_entity + args.alpha * loss_cluster + args.gmma * cross_cluster_loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    losses.append(loss.item())
                    e_c_losses.append(loss_entity.item())
                    c_losses.append(loss_cluster.item())
                    cross_cluster_losses.append(cross_cluster_loss.item())
                    pbar.set_description("Epoch: %d, Loss: %0.8f, E_C_loss: %0.8f, C_loss: %0.8f, cross_cluster_loss: %0.8f, lr: %0.6f" % (
                    epoch + 1, np.mean(losses), np.mean(e_c_losses), np.mean(c_losses), np.mean(cross_cluster_losses),optimizer.param_groups[0]['lr']))
            logging.info(
                "[Epoch %d/%d] [train loss: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses))
            )
        elif args.relation:
            model_entity_relation.train()
            model_relation.train()
            with tqdm(train_loader, desc="training") as pbar:
                losses = []
                e_r_losses = []
                r_losses = []
                cross_relaiton_losses = []
                for samples in pbar:
                    ids, lengths, \
                    source_entity, prev_outputs_entity, target_entity, mask_entity, \
                    source_cluster, prev_outputs_cluster, target_cluster, mask_cluster, \
                    source_relation, prev_outputs_relation, target_relation, mask_relation \
                        = samples['ids'], samples['lengths'], \
                          samples['source_entity'], samples['prev_outputs_entity'], samples['target_entity'], samples['mask_entity'],\
                          samples['source_cluster'], samples['prev_outputs_cluster'], samples['target_cluster'], samples['mask_cluster'],\
                          samples['source_relation'], samples['prev_outputs_relation'], samples['target_relation'], samples['mask_relation'],
                    optimizer.zero_grad()
                    loss_entity, entity_prob = model_entity_relation.get_loss(source_entity, prev_outputs_entity, target_entity, mask_entity)
                    loss_relation, relation_prob = model_relation.get_loss(source_relation, prev_outputs_relation, target_entity, mask_relation)
                    if args.loss == 1:
                        cross_relaiton_loss, _ = model_relation.get_cross_loss1(entity_prob)
                    elif args.loss == 2:
                        cross_relaiton_loss, _ = model_relation.get_cross_loss2(entity_prob)
                    elif args.loss == 3:
                        cross_relaiton_loss, _ = model_relation.get_cross_loss3(lengths, entity_prob)
                    loss =  loss_entity  + args.beta * loss_relation + args.theta * cross_relaiton_loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    losses.append(loss.item())
                    e_r_losses.append(loss_entity.item())
                    r_losses.append(loss_relation.item())
                    cross_relaiton_losses.append(cross_relaiton_loss.item())
                    pbar.set_description("Epoch: %d, Loss: %0.8f, E_R_loss: %0.8f, R_loss: %0.8f, cross_relation_loss: %0.8f, lr: %0.6f" % (
                    epoch + 1, np.mean(losses), np.mean(e_r_losses), np.mean(r_losses), np.mean(cross_relaiton_losses),optimizer.param_groups[0]['lr']))
            logging.info(
                "[Epoch %d/%d] [train loss: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses))
            )
        else:
            model_entity.train()
            with tqdm(train_loader, desc="training") as pbar:
                losses = []
                for samples in pbar:
                    ids, lengths, \
                    source_entity, prev_outputs_entity, target_entity, mask_entity, \
                    source_cluster, prev_outputs_cluster, target_cluster, mask_cluster, \
                    source_relation, prev_outputs_relation, target_relation, mask_relation \
                        = samples['ids'], samples['lengths'], \
                          samples['source_entity'], samples['prev_outputs_entity'], samples['target_entity'], samples['mask_entity'],\
                          samples['source_cluster'], samples['prev_outputs_cluster'], samples['target_cluster'], samples['mask_cluster'],\
                          samples['source_relation'], samples['prev_outputs_relation'], samples['target_relation'], samples['mask_relation'],
                    optimizer.zero_grad()
                    loss_entity, _ = model_entity.get_loss(source_entity, prev_outputs_entity, target_entity, mask_entity)
                    loss = loss_entity
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    steps += 1
                    losses.append(loss.item())
                    pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))
            logging.info("[Epoch %d/%d] [train loss: %f]" % (epoch + 1, args.num_epoch, np.mean(losses)))
        if args.valid:
            mrr_global = 0
            if ((epoch + 1) % args.save_interval == 0 and epoch != 0 and (epoch + 1) >= args.eval_begin) or (
                    epoch == args.num_epoch - 1):
                with torch.no_grad():
                    if args.cluster and args.relation:
                        mrr, hit1, hit3, hit10 = evaluate2(model_entity_cluster, model_entity_relation, valid_loader, device, args, train_valid, eval_valid, epoch+1)
                    elif args.cluster:
                        mrr, hit1, hit3, hit10 = evaluate(model_entity_cluster, valid_loader, device, args, train_valid, eval_valid)
                    elif args.relation:
                        mrr, hit1, hit3, hit10 = evaluate(model_entity_relation, valid_loader, device, args, train_valid, eval_valid)
                    else:
                        mrr, hit1, hit3, hit10 = evaluate(model_entity, valid_loader, device, args, train_valid, eval_valid)
                if mrr > mrr_global:
                    mrr_global = mrr
                    if args.cluster:
                        torch.save(model_entity_cluster.state_dict(), ckpt_path + "/ckpt-cluster.pt")
                    if args.relation:
                        torch.save(model_entity_relation.state_dict(), ckpt_path + "/ckpt-relation.pt")
                    if not args.cluster and not args.relation:
                        torch.save(model_entity_relation.state_dict(), ckpt_path + "/ckpt.pt")
        else:
            if ((epoch + 1) % args.save_interval == 0 and epoch != 0 and (epoch + 1) >= args.eval_begin) or (
                    epoch == args.num_epoch - 1):
                if args.cluster:
                    torch.save(model_entity_cluster.state_dict(), ckpt_path + "/ckpt-cluster.pt")
                if args.relation:
                    torch.save(model_entity_relation.state_dict(), ckpt_path + "/ckpt-relation.pt")
                if not args.cluster and not args.relation:
                    torch.save(model_entity.state_dict(), ckpt_path + "/ckpt.pt")
                with torch.no_grad():
                    if args.cluster and args.relation:
                        if args.eval_pattern == 0:
                            evaluate2(model_entity_cluster, model_entity_relation, test_loader, device, args,
                                      train_valid, eval_valid, epoch+1)
                        else:
                            evaluate3(model_entity_cluster, model_entity_relation, pre_test_loader, device, args,
                                      train_valid, eval_valid)
                    elif args.cluster:
                        evaluate(model_entity_cluster, test_loader, device, args,
                                  train_valid, eval_valid)
                    elif args.relation:
                        evaluate(model_entity_relation, test_loader, device, args,
                                  train_valid, eval_valid)
                    else:
                        evaluate(model_entity, test_loader, device, args,
                                  train_valid, eval_valid)
    if args.valid:
        if args.cluster and args.relation:
            model_entity_cluster = TransformerModel(args, train_set.dictionary, mode='entity')
            model_entity_relation = TransformerModel(args, train_set.dictionary, mode='entity')
            model_entity_cluster.load_state_dict(torch.load(os.path.join(ckpt_path, "ckpt-cluster.pt")))
            model_entity_relation.load_state_dict(torch.load(os.path.join(ckpt_path, "ckpt-relation.pt")))
            model_entity_cluster.args = args
            model_entity_cluster = model_entity_cluster.to(device)
            model_entity_relation.args = args
            model_entity_relation = model_entity_relation.to(device)
            with torch.no_grad():
                    evaluate2(model_entity_cluster, model_entity_relation, test_loader, device, args, train_valid, eval_valid)
        elif args.cluster:
            model_entity_cluster = TransformerModel(args, train_set.dictionary, mode='entity')
            model_entity_cluster.load_state_dict(torch.load(os.path.join(ckpt_path, "ckpt-cluster.pt")))
            model_entity_cluster.args = args
            model_entity_cluster = model_entity_cluster.to(device)
            with torch.no_grad():
                evaluate(model_entity_cluster, test_loader, device, args, train_valid, eval_valid)
        elif args.relation:
            model_entity_relation = TransformerModel(args, train_set.dictionary, mode='entity')
            model_entity_relation.load_state_dict(torch.load(os.path.join(ckpt_path, "ckpt-relation.pt")))
            model_entity_relation.args = args
            model_entity_relation = model_entity_relation.to(device)
            with torch.no_grad():
                evaluate(model_entity_relation, test_loader, device, args, train_valid, eval_valid)

def checkpoint(args):
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models_new', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(ckpt_path):
        print("Invalid path!")
        return
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/test.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    device = "cuda:{0}".format(args.cuda) if torch.cuda.is_available() else "cpu"
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", device=device, args=args)
    test_set = TestDataset(data_path=args.dataset + "/", device=device, src_file="test_triples.txt", args=args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    pre_test_set = TestDataset(data_path=args.dataset + "/", device=device, src_file="train_triples_rev.txt", args=args)
    pre_test_loader = DataLoader(pre_test_set, batch_size=args.test_batch_size, collate_fn=pre_test_set.collate_fn, shuffle=True)
    train_valid, eval_valid = train_set.get_next_valid()
    if args.cluster and args.relation:
        model_entity_cluster = TransformerModel(args, train_set.entity_dictionary, mode='entity')
        model_entity_cluster.load_state_dict(torch.load(os.path.join(ckpt_path, 'ckpt-cluster.pt'),  map_location=torch.device(device)))
        model_entity_cluster.args = args
        model_entity_cluster = model_entity_cluster.to(device)
        model_entity_relation = TransformerModel(args, train_set.entity_dictionary, mode='entity')
        model_entity_relation.load_state_dict(torch.load(os.path.join(ckpt_path, 'ckpt-relation.pt'), map_location=torch.device(device)))
        model_entity_relation.args = args
        model_entity_relation = model_entity_relation.to(device)
        with torch.no_grad():
            if args.eval_pattern == 0:
                evaluate2(model_entity_cluster, model_entity_relation, test_loader, device, args,
                          train_valid, eval_valid, epoch=-1)
            else:
                evaluate3(model_entity_cluster, model_entity_relation, pre_test_loader, device, args,
                          train_valid, eval_valid)
    else:
        model = TransformerModel(args, train_set.entity_dictionary,mode='entity')
        model.load_state_dict(torch.load(os.path.join(ckpt_path, 'ckpt-cluster.pt')))
        model.args = args
        model = model.to(device)
        with torch.no_grad():
            evaluate(model, test_loader, device, args, train_valid, eval_valid)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.test:
        checkpoint(args)
    else:
        train(args)
