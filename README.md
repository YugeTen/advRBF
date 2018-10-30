# RBF vs. adversarial example

### Goal
Utilising the nonlinear nature of RBF, investigate the benefit of replacing fully connected layer by RBF kernels in terms of:
1. Classification accuracy;
2. Robustness against adversarial attack.

---
### Usage
1. train a vanilla RBF classifier on CIFAR100 (Le net)
```
python main.py --mode train --dataset cifar-100 --D_out 100 --model_name vanilla_rbf
```
2. load trained classifier, keep training
```
python main.py --mode train --load_ckpt True --dataset cifar-100 --D_out 100 -model_name vanilla_rbf  
```
3. train a vanilla RBF classifier on Cat v Dog dataset (download from [link](https://www.kaggle.com/c/dogs-vs-cats/data), unzip and move _train_ and all its content into _advRBF/data_, rename _train_ as _catvdog_)
```
python main.py --mode train --load_ckpt True --dataset catvdog -model_name vanilla_rbf  
```
4. train vanilla RBF classifier and run adversarial attack using CIFAR10
```
python main.py --mode ftt --dataset CIFAR-10 --D_out 10 -model_name vanilla_rbf  
```

---
### Structure
#### root
_main.py_: parse arguments, call train/test/attack
_solver.py_: call trainable models (vanilla, vanilla_rbf, attack), run train/test/attack
#### ./models
_vanilla.py_, _vanilla_rbf.py_: define NNs
_rbf.py_: define RBF layer, called by _vanilla_rbf.py_
_attack.py_: define fgsm/i-fgsm
_data_loader.py_: load data for CIFAR10/CIAFAR100/catvdog

---
### Results
Below tables are the pre and post-attack accuracy and loss, measured at testing. All hyperparameters used are as default (eps = 0.02, 10 Gaussian kernels are used in place of _fc2_)

#### CIFAR10

|**Method**|Before Attack Accuracy (loss)|After Attack Accuracy (loss)|
| ------------- |:-------------:| -----:|
|Vanilla| 0.64 (1.082) | 0.19 (3.262)|
|RBF| 0.61 (1.088) | 0.16 (3.377)|

#### CIFAR100

|**Method**|Before Attack Accuracy (loss)|After Attack Accuracy (loss)|
| ------------- |:-------------:| -----:|
|Vanilla| 0.29 (3.021) | 0.10 (4.755)|
|RBF| 0.31 (2.882) | 0.05 (5.855)|

---
### Discussion
Using less parameters, RBF model beats baseline performance on CIFAR100 but not on CIFAR10, with a penalty of increased vulnerability to gradient-based adversarial attack (FGSM used).

Initial assessment is the Gaussians at _fc2_ smoothes the piecewise linear function of the vanilla CNN (which uses solely ReLu), and therefore makes it easier for white-box attack to trace the gradient and find efficient attacks.
<br>
