## Cold-start Multi-hop Reasoning by Hierarchical-guidance and Self-verification

This is the PyTorch implementation of of the **SelfHier** model for multi-hop reasoning.

## Overview
We present **SelfHier**, which utilizes the **self**-verification and **hier**archical-guidance to conduct multi-hop reasoning. Here is an overview of our model architecture:

![](figs/model.png)

## Training SelfHier
To reproduce our results or extend SelfHier model to more datasets, we provide a detailed description as follow:

1、Generating mapping files by running:
```
python utils.py --dataset NELL23K --gen-mapping --gen-eval-data
```

2、Then, our model utilizes [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) to mine logical rules. 

We provide a convenient script `run.sh` under `AnyBURL/` for mining and filtering high confidence rules (please modify the dataset name in `run.sh` and `config-learn.properties`). The above step helps generate `rule.dict` containing high quality rules under the dataset folder, then go to `data/` folder and run:
```
python utils.py --dataset NELL23K --gen-train-data --num 6 --out 6_rev_rule --max-len 3 --rule
```

3、go `data/NELL23K/` folder, generate the type_constrain.txt
```
python n-n.py
```
4、go `OpenKE-OpenKE-PyTorch/openke` folder
```
bash make.sh
```
5、go `OpenKE-OpenKE-PyTorch` folder, training embedding
```
python train_transe_NELL23K.py
```
6、go `SelfHier-main/` folder
```
python process_hierarchical_guidance --datapath 'data/NELL23K/'
```
7、training SlefHier
```
python train.py --cuda 0 --dataset NELL23K --batch-size 1024 --lr 5e-4 --dropout 0.1 --eval-begin 50 --num-epoch 50 --save-dir "model_4" --label-smooth 0.25 --encoder --save-interval 50 --l-punish --trainset "6_rev_rule" --prob 0.15 --beam-size 512 --test-batch-size 4
```
