# Introduction
Pytorch implementation from NUS DSML DSA5204 Group 12 for
* VPU ([A Variational Approach for Learning from Positive and Unlabeled Data][1])
* nnPU ([Positive-Unlabeled Learning with Non-Negative Risk Estimator][2])
* baseline model

# Method to run
1. install conda environment with yaml file or with following command:
	* `conda create -n DSA5204 python=3.8 pytorch=2.0.0 torchvision numpy scikit-learn -c pytorch`
2. conda activate DSA5204
3. python run.py --dataset pageblocks --batch_size 200 --method vpu --gpu 0

# Available options:
* --dataset {fashionMNIST, pageblocks}
* --batch_size
* --method {vpu, nnpu, raw}
* --gpu (*Valid only when running on cuda available machines*)
* --data_path
* --prior (*Only used for nnPU method)

[1]:https://arxiv.org/abs/1906.00642
[2]:https://arxiv.org/abs/1703.00593
