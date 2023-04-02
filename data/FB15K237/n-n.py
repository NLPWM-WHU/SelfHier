lef = {}
rig = {}
rellef = {}
relrig = {}

triple = open("train2id.txt", "r")
valid = open("valid2id.txt", "r")
test = open("test2id.txt", "r")

tot = (int)(triple.readline())
for i in range(tot):
	content = triple.readline()
	h,t,r = content.strip().split()
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

tot = (int)(valid.readline())
for i in range(tot):
	content = valid.readline()
	h,t,r = content.strip().split()
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

tot = (int)(test.readline())
for i in range(tot):
	content = test.readline()
	h,t,r = content.strip().split()
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

test.close()
valid.close()
triple.close()

f = open("type_constrain.txt", "w")
f.write("%d\n"%(len(rellef)))
for i in rellef:
	f.write("%s\t%d"%(i,len(rellef[i])))
	for j in rellef[i]:
		f.write("\t%s"%(j))
	f.write("\n")
	f.write("%s\t%d"%(i,len(relrig[i])))
	for j in relrig[i]:
		f.write("\t%s"%(j))
	f.write("\n")
f.close()
