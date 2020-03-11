# ARAE: Adversarially Robust Training of Autoencoders Improves Novelty Detection


This repository contains code to reproduce the main experiments of our paper "ARAE: Adversarially Robust Training of Autoencoders Improves Novelty Detection".

Also, you can find our pre-trained models and our final results.

![ARAE vs DAE](/MNIST-union/images/ARAEvsDAE.png)
*Unlike  DAE,  ARAE  that  is  trained  on  the  normal  class,  which  is  thedigit 8, reconstructs a normal instance when it is given an anomalous digit, from the class 1*

<!--
Here, we can provide the link to our paper, and we can write authors list.

<!--
This repository belongs to abnormal detection group in Sharif university of Technology. This project is under supervision of [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en) and is being conducted in [Data Science and Machine Learning Lab (DML)](http://dml.ir/) in Department of Computer Engineering. -->

<!--
The aim of the project is to learn a robust representation from normal samples in order to detect abnormality patterns. This work is mainly inspired by these papers, ["Adversarial examples for generative models"](https://arxiv.org/pdf/1702.06832.pdf) and ["Adversarial Manipulation of Deep Representations"](https://arxiv.org/pdf/1511.05122.pdf). More specifically, a new objective function is introduced by which an Autoencoder is trained so that it can both minimize pixel-wise error and learn a robust representation where it can capture variants of a sample in latesnt space. -->


## Prerequisites

* Tensorflow >= 1.15.0
* Keras >= 2.2.4
* torch >= 1.4


## Running the code

Having cloned the repository, you can reproduce our results:

### 1. L-inf model:

#### Preparing the data

At first, run prepare.py to prepare the data. The first argument to be passed is the dataset name. You may choose between fashion_mnist, mnist, and coil100.  For mnist and fashion_mnist, the next argument is the chosen protocol to prepare the data. For this argument you may choose betwen p1 and p2. If p2 is chosen, the next argument is the normal class number. Otherwise, the next argument is the anomaly percentage. Then you have to pass the class number.

Here are two examples for mnist and fashion_mnist datasets:

```
python3 prepare.py mnist p1 0.5 8
```
```
python3 prepare.py fashion_mnist p2 2
```

For the coil100 dataset, only the first protocol is available. After passing the dataset name, you have to pass the anomaly percentage. Next, you pass the number of normal classes <img src="https://latex.codecogs.com/gif.latex?\text n\in \{1,4,7\}" /> . After that, <img src="https://render.githubusercontent.com/render/math?math=n"> class numbers are passed.

Here is an example for the coil100 dataset:

```
python3 prepare.py 0.25 4 1 3 77 10
```

#### Training the model

#### Testing the model

### 2. Union model: ###

- At first, you need to complete the submit.sh according to your HPC setting to submit a sbatch file.
- Then, you just need to run main.sh on the HPC, It will automatically run code.py for 10 classes.

1. Note that by runnig the code.py you can save models and get results for a given class.
2. You can separately run code.py by passing a number of class as an argument.


## Citation
