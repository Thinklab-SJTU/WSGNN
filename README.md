# WSGNN

This is the official PyTorch implementation for the following KDD2022 paper:

Variational Inference for Training Graph Neural Networks in Low-Data Regime through Joint Structure-Label Estimation

Danning Lao\*, Xinyu Yang\*, Qitian Wu, and Junchi Yan.

(*: equal contribution)

## Installation

```bash
conda create -n WSGNN python=3.8
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit=11.0
pip install scipy
pip install --no-index torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric==2.0.2
```

## Run the code

**WSGNN**

* Cora

  `python main.py --dataset cora --lr 0.01 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.5 --graph_skip_conn 0.5 --graph_learn_num_pers 8 --no_bn --neg_sampling_ratio 5 --epoch 100 --hops 4 --display_step 50`

* Citeseer

  `python main.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.5 --graph_skip_conn 0.5 --graph_learn_num_pers 8 --no_bn --neg_sampling_ratio 5 --epoch 200 --hops 6 --display_step 50`

* Pubmed

  `python main.py --dataset pubmed --lr 0.01 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.5 ---graph_skip_conn 0.5 --graph_learn_num_pers 8 --no_bn --neg_sampling_ratio 5 --epoch 200 --hops 4 --display_step 50`

* Disease-NC

  `python nc_main.py --dataset disease_nc --lr 0.001 --weight_decay 5e-4 --hidden_channels 64 --dropout 0.5  --graph_learn_num_pers 8 --neg_sampling_ratio 5 --hops 3 --display_step 100 --epoch 1000`

* Disease-LP

  `python lp_main.py --dataset disease_lp --lr 0.01 --weight_decay 0 --hidden_channels 64 --dropout 0.5 --graph_learn_num_pers 1 --hops 1 --no_bn --epoch 1000 --display_step 100 --neg_sampling_ratio 5 `

**GCN**

* Cora

  `python baseline.py --dataset cora --method gcn --lr 0.005 --weight_decay 5e-4 --hidden_channels 8 --num_layers 1  --dropout 0.6 --no_bn --neg_sampling_ratio 5 --epochs 1000 --display_step 100`

* Citeseer

  `python baseline.py --dataset citeseer --method gcn --lr 0.005 --weight_decay 5e-4 --hidden_channels 8 --num_layers 1  --dropout 0.6 --no_bn --neg_sampling_ratio 5 --epochs 3000 --display_step 100`

* Pubmed

  `python baseline.py --dataset pubmed --method gcn --lr 0.005 --weight_decay 5e-4 --hidden_channels 8 --num_layers 1  --dropout 0.6 --no_bn --neg_sampling_ratio 5 --epochs 1000 --display_step 100`

* Disease-NC

  `python nc_baseline.py --dataset disease_nc --method gcn --hidden_channels 8 --num_layers 1 --lr 0.001 --weight_decay 5e-4 --dropout 0.6 --no_bn --display_step 100 --epochs 1000 --lambda2 0`

* Disease-LP

  `python lp_baseline.py --dataset disease_lp --method gcn --hidden_channels 8 --num_layers 1 --lr 0.01 --weight_decay 0 --dropout 0.6 --no_bn --display_step 100 --epochs 1000 --neg_sampling_ratio 5 --lambda1 0`

**GAT**

* Cora

  `python baseline.py --dataset cora --method gat --lr 0.005 --weight_decay 5e-4 --hidden_channels 8 --num_layers 1 --dropout 0.6 --no_bn --neg_sampling_ratio 5 --epochs 1000 --display_step 100  `

* Citeseer

  `python baseline.py --dataset citeseer --method gat --lr 0.005 --weight_decay 5e-4 --hidden_channels 8 --out_heads 8 --num_layers 1 --dropout 0.6 --no_bn --neg_sampling_ratio 5 --epochs 1000 --display_step 100`

* Pubmed

  `python baseline.py --dataset pubmed --method gat --lr 0.005 --weight_decay 5e-4 --hidden_channels 8 --out_heads 8 --num_layers 1 --dropout 0.6 --no_bn --neg_sampling_ratio 5 --epochs 1000 --display_step 100`

* Disease-NC

  `python nc_baseline.py --dataset disease_nc --method gat --hidden_channels 8 --num_layers 1 --lr 0.001 --weight_decay 5e-4 --dropout 0.6 --no_bn --display_step 100 --epochs 1000 --lambda2 0`

* Disease-LP

  `python lp_baseline.py --dataset disease_lp --method gat --hidden_channels 8 --num_layers 1 --lr 0.01 --weight_decay 0 --dropout 0.6 --no_bn --display_step 100 --epochs 1000 --neg_sampling_ratio 5 --lambda1 0`

## Citation

If WSGNN is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```
@inproceedings{lao2022wsgnn,
    title={Variational Inference for Training Graph Neural Networks in Low-Data Regime through Joint Structure-Label Estimation},
    author={Lao, Danning and Yang, Xinyu, and Wu, Qitian and Yan, Junchi},
    booktitle={KDD},
    year={2022}
}
```

## Reference code

1. PyG (PyTorch Geometric): https://github.com/pyg-team/pytorch_geometric
1. Non-Homophily-Benchmarks:  https://github.com/CUAI/Non-Homophily-Benchmarks
1. IDGL: https://github.com/hugochan/IDGL
