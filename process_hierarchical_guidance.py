import os
import argparse
from sklearn.cluster import KMeans
import torch
import numpy as np

def K_means_clustering(K, datapath='./data/FB15K237-20/'):
	cluste2idPath = os.path.join(datapath, 'cluster2id' + str(K) + '.txt')
	entity2idPath = os.path.join(datapath, 'entity2id.txt')
	relation2idPath = os.path.join(datapath, 'relation2id.txt')
	pretrained_emb_file = os.path.join(datapath, 'entity2vec.bern')
	with open(cluste2idPath, "w") as f:
		for cluster in range(K):
			f.write('C'+str(cluster) + '\t' + str(cluster) + '\n')
	# read entity IDs
	with open(entity2idPath, "r") as f:
		entity2id = {}
		for line in f:
			try:
				entity, eid = line.split()
				entity2id[entity] = int(eid)
			except:
				continue
	# read relation IDs
	with open(relation2idPath, "r") as f:
		relation2id = {}
		for line in f:
			try:
				relation, rid = line.split()
				relation2id[relation] = int(rid)
			except:
				continue

	# read entity embeddings
	entity2emb = []
	with open(pretrained_emb_file, "r") as f:
		for line in f:
			entity2emb.append([float(value) for value in line.split()])

	# entity2emb = np.load(pretrained_emb_file)
	# entity2emb = list(entity2emb)

	# K Means CLustering
	kmeans_entity = KMeans(n_clusters=K, random_state=0).fit(entity2emb)

	cluster_embeddings = kmeans_entity.cluster_centers_
	culster_emb_file = datapath+'cluster2vec'+str(K)+'.bern'
	with open(culster_emb_file, 'w') as f:
		for i in cluster_embeddings:
			for j in i:
				f.writelines(str(j) + ' ')
			f.writelines('\n')

	# assign cluster label to entities
	entity2cluster = {}

	for idx, label in enumerate(kmeans_entity.labels_):
		entity2cluster[idx] = int(label)

	# print(entity2cluster)
	save_path = 'entity2clusterid_' + str(K) +'.txt'

	ent2clusterFile = os.path.join(datapath, save_path)
	with open(ent2clusterFile, 'w') as f:
		for ent in entity2cluster.keys():
			f.write(str(ent) + '\t' + str(entity2cluster[ent]) + '\n')

def generate_cluster(K, datapath = './data/FB15K237-20/'):
	in_pathdata = datapath + '/in_6_rev_rule.txt'
	out_pathdata = datapath + '/out_6_rev_rule.txt'
	e2cpath = datapath+'entity2clusterid_' + str(K) + '.txt'
	e2c_dict={}
	with open(e2cpath, 'r') as f:
		for line in f:
			e,c = line.strip().split('\t')
			e2c_dict[e]='C'+ str(c)

	in_path_list_all = []
	with open(in_pathdata, 'r') as f:
		for line in f:
			path_list = line.split()
			path_list[0] = e2c_dict[path_list[0]]
			new_path = ' '.join(path_list)
			in_path_list_all.append(new_path)

	in_new_data = datapath + 'in_6_rev_rule_cluster' + str(K) + '.txt'
	with open(in_new_data, 'w') as f:
		for path in in_path_list_all:
			f.write(str(path) + '\n')

	out_path_list_all = []
	with open(out_pathdata, 'r') as f:
		for line in f:
			path_list = line.split()
			length = len(path_list)
			entity_num = int(length/2)
			n = 1
			for i in range(entity_num):
				path_list[n]=e2c_dict[path_list[n]]
				n +=2
			new_path = ' '.join(path_list)
			out_path_list_all.append(new_path)

	out_new_data = datapath + 'out_6_rev_rule_cluster' + str(K) + '.txt'
	with open(out_new_data, 'w') as f:
		for path in out_path_list_all:
			f.write(str(path) + '\n')

def generate_relation(R, datapath = './data/FB15K237-20/'):
	relation2emb = []
	entity2emb = []
	with open(datapath + 'relation2vec.bern', "r") as f:
		for line in f:
			relation2emb.append([float(value) for value in line.split()])
	relation2emb = torch.tensor(relation2emb)
	relation2emb = torch.cat((relation2emb, -relation2emb))

	with open(datapath + 'entity2vec.bern', "r") as f:
		for line in f:
			entity2emb.append([float(value) for value in line.split()])
	entity2emb = torch.tensor(entity2emb)

	paths = []
	with open(datapath + 'in_6_rev_rule.txt', "r") as f:
		for line in f:
			paths.append([line.strip().split()[0]])
	with open(datapath + 'out_6_rev_rule.txt', "r") as f:
		for i, line in enumerate(f):
			paths[i].extend(line.strip().split())
	print(paths[0])

	R_cos_dis = [[] for i in range(relation2emb.size()[0])]
	for path in paths:
		for i in range(int((len(path) - 1) / 2)):
			e1 = entity2emb[int(path[2 * i])]
			r = relation2emb[int(path[2 * i + 1].strip('R'))]
			e2 = entity2emb[int(path[2 * i + 2])]
			cos_dis = float(1 - torch.cosine_similarity(r, e2 - e1, dim=0))
			# if cos_dis == 1.0:
			#     aaaa = 1111
			R_cos_dis[int(path[2 * i + 1].strip('R'))].extend([cos_dis])

	print(R_cos_dis[220])

	R_distribution = []
	for R_dis in R_cos_dis:
		if R_dis == []: #这个地方是为了在冷启动实验中没有出现的r
			R_distribution.append([])
		else:
			max_dis = max(R_dis)
			min_dis = min(R_dis)
			if min_dis != max_dis:
				#R_distribution.append(np.arange(min_dis, max_dis + (max_dis - min_dis) / R, (max_dis - min_dis) / R))
				#R_distribution.append([min_dis: max_dis + (max_dis - min_dis) / R: (max_dis - min_dis) / R])
				R_distribution.append([(min_dis + (max_dis - min_dis) / R * i )for i in range(R+1)])
			else:
				R_distribution.append([min_dis, max_dis])

	new_R = []
	for i, r_distri in enumerate(R_distribution):
		for j in range(R):
			new_R.append('R{0}'.format(i) + '_{0}'.format(j))

	with open(datapath + 'multi_relation2id.txt', "w") as f:
		for i, rel in enumerate(new_R):
			f.write(str(rel) + '\t' + str(i) + '\n')

	new_path = []
	for iii, path in enumerate(paths):
		if iii == 606156:
			print('11111')
		new_path.append([])
		for i in range(int((len(path) - 1) / 2)):
			start = path[2 * i]
			rel = path[2 * i + 1]
			end = path[2 * i + 2]

			e1 = entity2emb[int(start)]
			r = relation2emb[int(rel.strip('R'))]
			e2 = entity2emb[int(end)]

			cos_dis = float(1 - torch.cosine_similarity(r, e2 - e1, dim=0))
			distribution = R_distribution[int(rel.strip('R'))]
			if len(distribution) == 2:
				new_path[-1].extend([rel])
			else:
				if cos_dis == distribution[-1]:
					new_path[-1].extend([new_R[int(rel.strip('R')) * R + R - 1]])
				else:
					for j in range(len(distribution) - 1):
						if cos_dis >= distribution[j] and cos_dis < distribution[j + 1]:
							new_path[-1].extend([new_R[int(rel.strip('R')) * R + j]])
			new_path[-1].extend([end])

	out_new_data = datapath + 'out_6_rev_rule_relation.txt'
	with open(out_new_data, 'w') as f:
		for path in new_path:
			path = ' '.join(path).strip()
			f.write(str(path) + '\n')

def set_pre_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--K', default=50, type=int)
	parser.add_argument('--R', default=3, type=int)
	parser.add_argument('--datapath', default='data/FB15K237/', type=str)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = set_pre_args()
	K_means_clustering(K = args.K,
					   datapath=args.datapath,
					   )
	generate_cluster(K = args.K,
					   datapath=args.datapath,
					   )
	generate_relation(R = args.R,
					  datapath=args.datapath)
