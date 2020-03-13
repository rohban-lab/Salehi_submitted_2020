# ARAE: Adversarially Robust Training of Autoencoders Improves Novelty Detection


This repository contains code to reproduce the main experiments of our paper "ARAE: Adversarially Robust Training of Autoencoders Improves Novelty Detection".

Also, you can find our pre-trained models and our final results.

![ARAE vs DAE](/MNIST-union/images/ARAEvsDAE.png)
*Unlike  DAE,  ARAE  that  is  trained  on  the  normal  class,  which  is  the digit 8, reconstructs a normal instance when it is given an anomalous digit, from the class 1.*

<!--
Here, we can provide the link to our paper, and we can write authors list.

<!--
This repository belongs to abnormal detection group in Sharif university of Technology. This project is under supervision of [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en) and is being conducted in [Data Science and Machine Learning Lab (DML)](http://dml.ir/) in Department of Computer Engineering. -->

<!--
The aim of the project is to learn a robust representation from normal samples in order to detect abnormality patterns. This work is mainly inspired by these papers, ["Adversarial examples for generative models"](https://arxiv.org/pdf/1702.06832.pdf) and ["Adversarial Manipulation of Deep Representations"](https://arxiv.org/pdf/1511.05122.pdf). More specifically, a new objective function is introduced by which an Autoencoder is trained so that it can both minimize pixel-wise error and learn a robust representation where it can capture variants of a sample in latesnt space. -->

## Running the code

Having cloned the repository, you can reproduce our results:

### 1. L-inf model:

If you want to use the pre-trained models, skip to [this section](https://github.com/rohban-lab/Salehi_submitted_2020#testing).

#### Preparing the data

At first, run prepare.py to prepare the data. The first argument to be passed is the dataset name. You may choose between fashion_mnist, mnist, and coil100.  For mnist and fashion_mnist, the next argument is the chosen protocol to prepare the data. For this argument, you may choose between p1 and p2. If p2 is chosen, the next argument is the normal class number. Otherwise, the next argument is the anomaly percentage. Then you have to pass the class number.

Here are two examples for mnist and fashion_mnist datasets:

```
python3 prepare.py mnist p1 0.5 8
```
```
python3 prepare.py fashion_mnist p2 2
```

For the coil100 dataset, only the first protocol is available. After passing the dataset name, you have to pass the anomaly percentage. Next, you pass the number of normal classes. After that, the class numbers are passed.

Here is an example for the coil100 dataset:

```
python3 prepare.py coil100 0.25 4 1 3 77 10
```

#### Training

To train the model by yourself, you have to run the following script:

```
python3 train.py
```

#### Testing

If you trained the model yourself, you can use the following script to test your model:

```
python3 test.py
```

To use the pre-trained models, you have to pass the model directory to test.py. The pre-trained models are available in the pretrained_models folder. For mnist, the model is trained using both protocols, for all classes. For fashion_mnist, the model is trained using the second protocol for all classes. Finally, for coil100, 30 models are trained by varying the number of normal classes between 1, 4, and 7. To test the first protocol for mnist, you have to pass the anomaly percentage as an additional argument.

Here are examples for all the three datasets:

```
python3 test.py ./pretrained_models/mnist_pretrained/p2/8/
```
```
python3 test.py ./pretrained_models/mnist_pretrained/p1/5/ 0.5
```
```
python3 test.py ./pretrained_models/fashion_mnist_pretrained/2/
```
```
python3 test.py ./pretrained_models/coil100_pretrained/4/30/
```

### 2. Union model:

- At first, you need to complete the submit.sh according to your HPC setting to submit a sbatch file.
- Then, you just need to run main.sh on the HPC, It will automatically run code.py for 10 classes.

1. Note that by runnig the code.py you can save models and get results for a given class.
2. You can separately run code.py by passing a number of class as an argument.


## Citation
