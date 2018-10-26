# advRBF
RBF vs. adversarial example

### Usage
1. train a vanilla RBF classifier on CIFAR100 (Le net)
```
python main.py --mode train --dataset cifar-100 --model_name vanilla_rbf
```
2. load trained classifier, keep training
```
python main.py --mode train --load_ckpt True --dataset cifar-100 -model_name vanilla_rbf  
```

<br>
