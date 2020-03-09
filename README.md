# ARAE: Adversarially Robust Training of Autoencoders Improves Novelty Detection


This repository contains code to reproduce the main experiments of our paper "ARAE: Adversarially Robust Training of Autoencoders Improves Novelty Detection".

Also, you can find our pre-trained models and our final results.

![ARAE vs DAE](/MNIST-union/images/ARAEvsDAE.png)
Format: ![Alt Text](url)

<!--
Here, we can provide the link to our paper, and we can write authors list.

<!--
This repository belongs to abnormal detection group in Sharif university of Technology. This project is under supervision of [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en) and is being conducted in [Data Science and Machine Learning Lab (DML)](http://dml.ir/) in Department of Computer Engineering. -->

<!--
The aim of the project is to learn a robust representation from normal samples in order to detect abnormality patterns. This work is mainly inspired by these papers, ["Adversarial examples for generative models"](https://arxiv.org/pdf/1702.06832.pdf) and ["Adversarial Manipulation of Deep Representations"](https://arxiv.org/pdf/1511.05122.pdf). More specifically, a new objective function is introduced by which an Autoencoder is trained so that it can both minimize pixel-wise error and learn a robust representation where it can capture variants of a sample in latesnt space. -->


## Prerequisites

* Tensorflwo 1.12
* Keras 2.2.4
* torch 1.4


## Running the code

Having cloned the repository,

### 1. L-inf model: ###

the script can be run using the following arguments:

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

Note: To submit a task on HPC, all you need to do is to call run.sh. It will be run v1.py using HPC's GPU.

```
qsub run.sh
```

### 2. Union model: ###

- At first, you need to complete the submit.sh according to your HPC setting to submit a sbatch file.
- Then, you just need to run main.sh on the HPC, It will automatically run code.py for 10 classes.

1. Note that by runnig the code.py you can save models and get results for a given class.
2. Note that you can separately run code.py by passing a number of class as an argument.


## Citation
