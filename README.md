Pytorch implementation for
* VPU ([A Variational Approach for Learning from Positive and Unlabeled Data][1])
* nnPU ([Positive-Unlabeled Learning with Non-Negative Risk Estimator][2])
* baseline model

# Method to run
1. install conda environment with yaml file
2. python run.py --dataset fashionMNIST --batch_size 500 --method vpu --gpu 0

Available options:
* --dataset {fashionMNIST, pageblocks}
* --batch_size
* --method {vpu, nnpu, raw}
* --gpu (*Only valid when running on cuda available machines*)
* --datapath

[1]:https://arxiv.org/abs/1906.00642
[2]:https://arxiv.org/abs/1703.00593
