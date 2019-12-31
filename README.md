# Enhanced-Abnormal-Detection-Using-Adverserial-Attacks
This repository belongs to abnormal detection group in Sharif university of Technology. This project is under supervision of [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en) and is being conducted in [Data Science and Machine Learning Lab (DML)](http://dml.ir/) in Department of Computer Engineering.

The aim of the project is to learn a robust representation from normal samples in order to detect abnormality patterns. This work is mainly inspired by these papers, ["Adversarial examples for generative models"](https://arxiv.org/pdf/1702.06832.pdf) and ["Adversarial Manipulation of Deep Representations"](https://arxiv.org/pdf/1511.05122.pdf). More specifically, a new objective function is introduced by which an Autoencoder is trained so that it can both minimize pixel-wise error and learn a robust representation where it can capture variants of a sample in latesnt space.

## Prerequisites

* Tensorflwo 1.12
* Keras 2.2.4
* torch 1.3.1



## Getting Started

Having cloned the repository, the script can be run using the following arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  -g, --gpu_id          determines gpu id
  -d, --data_path       path to dataset
  -c, --checkpoint_path the address in which the model is going to be saved.
  -n, --class_number    choose normal class
  -l, --learning_rate   learning rate
  -s, --num_steps       number of epochs
  -b, --batch_size      mini batch size
  -k, --coef            setting coeficient in error function to control the effect of adverserial attack
  -e, --kernel_size     kernel size
  -t, --stride          kernel stride
```

Having decided about the arguments, the script can be run like below:

```
python v1.py --gpu_id "1" -l 0.0001
```
