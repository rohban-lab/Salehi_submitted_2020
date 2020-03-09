#!/usr/bin/python

import sys


import numpy as np


import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


import itertools



# Hyperparams.

FLAGS = {
"normal_class" : int(sys.argv[1]),
"batch_size": 128,
"learning_rate" : 0.001,
"n_epoch": 150}


# Reading data.

def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)

    return label_indices


transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
indices = get_same_index(trainset.train_labels, FLAGS['normal_class'])
dataloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS['batch_size'],
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=1)

indices2 = get_same_index(testset.test_labels, FLAGS['normal_class'])
validationloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2),
                                         num_workers=1)


# Model.

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        res = decoded.view(-1, 1, 28, 28)
        return res


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.0)


autoencoder = AutoEncoder()
autoencoder.apply(init_weights)


# Loss and Optimizer.

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=FLAGS['learning_rate'])


# GPU!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)


# L2 attack.

def pgd_attack(model, images, eps=5, alpha=0.1, iters=100):
    
    images = images.to(device).view(-1, 28 * 28)

    loss = torch.nn.MSELoss()
    
    ori_images = images

    images = images + (torch.rand_like(images) - 0.5).renorm(p=2, dim=0, maxnorm=eps)
    images = torch.clamp(images, min=0, max=1)


    for i in range(iters):
        
        images.requires_grad = True

        base = model.encoder(ori_images.detach())
        outputs = model.encoder(images)

        model.zero_grad()
        cost = loss(outputs, base).to(device)
        cost.backward()

        g = images.grad.data
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*(len(images.shape) - 1)))
        # g_norm = g.view(g.shape[0], -1).norm(dim=1)[:,None,None,None]
        scaled_g = g / (g_norm + 1e-10)

        adv_images = images + alpha*scaled_g
        eta = (adv_images - ori_images).renorm(p=2, dim=0, maxnorm=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
    return images.view(-1, 1, 28, 28)


# L-inf attack.

def pgd_attack1(model, images, eps=0.2, alpha=0.01, iters=40):
    
    images = images.to(device).view(-1, 28 * 28)

    loss = torch.nn.MSELoss()
    
    ori_images = images

    images = images + (torch.rand_like(images) - 0.5) * 2 * eps
    # images = images + torch.zeros_like(images).uniform_(-eps, eps)
    images = torch.clamp(images, min=0, max=1)


    for i in range(iters):
        
        images.requires_grad = True

        base = model.encoder(ori_images.detach())
        outputs = model.encoder(images)

        model.zero_grad()
        cost = loss(outputs, base).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
    return images.view(-1, 1, 28, 28)


# Spatial (Rotation and translation) attack.

def spatial_attack(model, images, rot=30, tr=4/28):

    bs = images.size(0)

    rots = torch.linspace(-rot, rot, steps=int(rot) + 1)
    trans = torch.linspace(-tr, tr, steps=3)
    tfms = torch.tensor(list(itertools.product(rots, trans, trans))).cuda(device=device)
    all_rots = tfms[:, 0]
    all_trans = tfms[:, 1:]

    ntfm = all_rots.shape[0]

    transformed = transform_input(images.repeat_interleave(repeats=ntfm, dim=0), all_rots.repeat([bs]), all_trans.repeat([bs, 1]))

    base = model.encoder(images.repeat_interleave(repeats=ntfm, dim=0).view(-1, 28 * 28).detach())
    outputs = model.encoder(transformed.view(-1, 28 * 28).detach())

    cost = torch.mean((outputs - base) ** 2, dim=1).to(device)
    maximum = torch.argmax(cost.view(bs, ntfm), dim=1)
    index = maximum + torch.tensor([i * ntfm for i in range(bs)]).to(device)

    res = transformed.view(bs * ntfm, 28 * 28)[index, :]
    return res.view(bs, 1, 28, 28)


_MESHGRIDS = {}

def make_meshgrid(x):
    bs, _, _, dim = x.shape
    device = x.get_device()

    key = (dim, bs, device)
    if key in _MESHGRIDS:
        return _MESHGRIDS[key]

    space = torch.linspace(-1, 1, dim)
    meshgrid = torch.meshgrid([space, space])
    gridder = torch.cat([meshgrid[1][..., None], meshgrid[0][..., None]], dim=2)
    grid = gridder[None, ...].repeat(bs, 1, 1, 1)
    ones = torch.ones(grid.shape[:3] + (1,))
    final_grid = torch.cat([grid, ones], dim=3)
    expanded_grid = final_grid[..., None].cuda()

    _MESHGRIDS[key] = expanded_grid

    return expanded_grid

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return torch.cuda.FloatTensor(size=size).uniform_(**args)

def make_slice(a, b, c):
    to_cat = [a[None, ...], b[None, ...], c[None, ...]]
    return torch.cat(to_cat, dim=0)

def make_mats(rots, txs):
    # rots: degrees
    # txs: % of image dim

    rots = rots * 0.01745327778 # deg to rad
    txs = txs * 2

    cosses = torch.cos(rots)
    sins = torch.sin(rots)

    top_slice = make_slice(cosses, -sins, txs[:, 0])[None, ...].permute([2, 0, 1])
    bot_slice = make_slice(sins, cosses, txs[:, 1])[None, ...].permute([2, 0, 1])

    mats = torch.cat([top_slice, bot_slice], dim=1)

    mats = mats[:, None, None, :, :]
    mats = mats.repeat(1, 28, 28, 1, 1)
    return mats

def transform_input(x, rots, txs):
    assert x.shape[2] == x.shape[3]

    with torch.no_grad():
        meshgrid = make_meshgrid(x)
        tfm_mats = make_mats(rots, txs)

        new_coords = torch.matmul(tfm_mats, meshgrid)
        new_coords = new_coords.squeeze_(-1)

        new_image = F.grid_sample(x, new_coords, align_corners=True)
        
    return new_image


# Plotting and some other main functions.

def imshow(img):
    img = img.to('cpu')
    npimg = img.numpy()
    plt.imshow(npimg.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.show()

def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 28, 28)
    return x

def show(image_batch, rows=1):

    # Set plot dimensions
    cols = np.ceil(image_batch.shape[0] / rows)
    plt.rcParams['figure.figsize'] = (0.0 + cols, 0.0 + rows) # set default size of plots
    
    for i in range(image_batch.shape[0]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_batch[i], cmap="gray")
        plt.axis('off')
    plt.show()

def show_process_for_trainortest(input_img, recons_img, attacked_img=None, train=True, attack=False):

    if input_img.shape[0] > 15:
        n = 15
    else:
        n = input_img.shape[0]
        
    if train:
        print("Inputs:")
        show(input_img[0:n].view((1,-1,28,28))[0].cpu())
        # Calculate reconstructions
        if attack:
            print("Inputs after attack:")
            show(attacked_img[0:n].view((1,-1,28,28))[0].cpu().detach().numpy())     
        print("Reconstructions:")
        show(recons_img[0:n].view((1,-1,28,28))[0].cpu().detach().numpy())
    else:
        print("Test Inputs:")
        show(input_img[0:n].view((1,-1,28,28))[0].cpu())
        # Calculate reconstructions
        print("Test Reconstructions:")
        show(recons_img[0:n].view((1,-1,28,28))[0].cpu().detach().numpy())


def test():

    plt.rcParams['figure.figsize'] = (5, 5)

    loss_epoch = 0.0
    n_batches = 0
    label_score = []

    autoencoder.eval()

    with torch.no_grad():
        for data in testloader:

            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = autoencoder(inputs)

            # pic1 = to_img(inputs.cpu().data)
            # pic2 = to_img(outputs.cpu().data)
            # show_process_for_trainortest(pic1, pic2)

            outputs = outputs.view(outputs.shape[0], -1)
            inputs = inputs.view(inputs.shape[0], -1)
            scores = torch.mean((outputs - inputs) ** 2, dim = 1)
            # print(scores)

            # Save triple of (idx, label, score) in a list.
            label_score += list(zip(labels.cpu().data.numpy().tolist(), scores.cpu().data.numpy().tolist()))
            n_batches += 1


    labels, scores = zip(*label_score)
    labels = np.array(labels)

    indx1 = labels == FLAGS['normal_class']
    indx2 = labels != FLAGS['normal_class']
    labels[indx1] = 1
    labels[indx2] = 0

    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)

    print('auc: %.5f' % (roc_auc))

    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #         lw=lw, label='ROC curve (area = %f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    autoencoder.train()


def validation():

    autoencoder.eval()

    total = 0

    with torch.no_grad():

        for data in validationloader:

            inputs, _ = data[0].to(device), data[1].to(device)

            outputs = autoencoder(inputs)

            outputs = outputs.view(-1, 28 * 28)
            inputs = inputs.view(-1, 28 * 28)

            scores = torch.mean((outputs - inputs) ** 2, dim = 1)
            total += torch.sum(scores).item()

    autoencoder.train()

    return total


# Training loop.

print(FLAGS['normal_class'])
print(FLAGS['normal_class'])


autoencoder.train()

num_epchos = 500

for epoch in range(num_epchos):
    
    steps = 0
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        
        inputs, _ = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        
        adv1 = pgd_attack(autoencoder, inputs)
        adv2 = pgd_attack1(autoencoder, inputs)
        adv3 = spatial_attack(autoencoder, inputs)
        adv = torch.cat([adv1, adv2, adv3], dim=0)

        outputs = autoencoder(adv)

        base = autoencoder.encoder(torch.cat([inputs, inputs, inputs], dim=0).view(-1, 28 * 28))
        outputs2 = autoencoder.encoder(adv.view(-1, 28 * 28))
        
        loss = criterion(torch.cat([inputs, inputs, inputs], dim=0), outputs) + (0.2) * criterion(base, outputs2)
        
        loss.backward()
        optimizer.step()
        
        steps += 1
        running_loss += loss.item()

    # print statistics.
    print('%d loss: %.5f' % (epoch + 1, running_loss / steps))

    print('%d validation loss: %.5f' % (epoch + 1, validation()))

    test()
    
    # Saving the model.
    if epoch % 50 == 0:
        PATH = './model' + str(sys.argv[1]) + '-' + str(epoch + 1) + '.pth'
        torch.save(autoencoder.state_dict(), PATH)
	
    # Plot pics.
    # if epoch % 5 == 0:
    #   pic1 = to_img(inputs.cpu().data)
    #   pic2 = to_img(outputs.cpu().data)
    #   pic3 = to_img(adv.cpu().data)
    #   show_process_for_trainortest(pic1, pic2, pic3, train=True, attack=True)

print('Finished Training')


