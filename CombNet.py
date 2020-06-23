import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import os


class DataGen(Dataset):
    def __init__(self):
        self.base_dir = 'datasets'
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.imgs = []
        self.target = [[0 for i in range(7)] for j in range(115 * 7)]
        for c in self.classes:
            self.imgs += os.listdir(os.path.join(self.base_dir, c))
        for i in range(7 * 115):
            self.target[i][int(i / 115)] = 1

        # define transform
        self.transform = transforms.Compose([ToTensor()])

        # random indexing
        rnd_index = np.random.permutation(len(self.imgs)).astype(np.int)
        self.imgs = np.array(self.imgs)[rnd_index]
        self.target = np.array(self.target)[rnd_index]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        c = self.classes[np.argmax(self.target[index])]
        img_path = os.path.join(self.base_dir, c, self.imgs[index])
        img = Image.open(img_path).convert("RGB")
        # image crop
        img = img.crop((75, 0, 525, 450))
        # resize
        img = img.resize((224, 224))

        one_img, one_target = self.transform((np.array(img), self.target[index]))

        return one_img, one_target


class ToTensor(object):

    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, data):
        img = data[0]
        target = data[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img), torch.from_numpy(target)


d = DataGen()
indices = torch.randperm(len(d)).tolist()
dataset_train = torch.utils.data.Subset(d, indices[:600])
dataset_test = torch.utils.data.Subset(d, indices[600:])
train_loader = DataLoader(dataset_train, batch_size=60, num_workers=1)
test_loader = DataLoader(dataset_test, batch_size=1, num_workers=1)


class FeaturePyramidStructure(nn.Module):

    """ CombNet's backbone (using ResNet-18) """

    def __init__(self):
        super(FeaturePyramidStructure, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.layer_list = list(self.resnet18.children())
        self.c1_module = self.layer_list[:3]
        self.c5_module = self.layer_list[3:5]
        self.c17_module = self.layer_list[5:9]
        self.fc = nn.Linear(in_features=512, out_features=7)

    def forward(self, x):
        step = x

        # conv1, batchnorm, relu
        for layer in self.c1_module:
            step = layer(step)
        c1 = step

        # maxpool, conv2_x
        for layer in self.c5_module:
            step = layer(step)
        c5 = step

        # conv3_x ~ conv5_x, avgpool
        for layer in self.c17_module:
            step = layer(step)
        c17 = step

        top_down_1 = F.softmax(self.fc(c17.view(c17.shape[0], -1)), dim=1)

        return top_down_1, c5, c1

    # top_down_1 is backbone's output
    # c5 goes to SubNet2
    # c1 goes to SubNet1


class SubNet1(nn.Module):

    """ Backbone's sub-network1 for other scale features  """

    def __init__(self):
        super(SubNet1, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        c1_1 = F.relu(self.conv1(self.pool(x)))  # (112, 112, 64) -> (56, 56, 128)
        c1_2 = F.relu(self.conv1_2(c1_1))  # (56, 56, 128) -> (56, 56, 128)
        c2_1 = F.relu(self.conv2(self.pool(c1_2)))  # (56, 56, 128) -> (28, 28, 256)
        c2_2 = F.relu(self.conv2_2(c2_1))  # (28, 28, 256) -> (28, 28, 256)
        c3_1 = F.relu(self.conv3(self.pool(c2_2)))  # (28, 28, 256) -> (14, 14, 256)
        c3_2 = F.relu(self.conv4(c3_1))  # (14, 14, 256) -> (14, 14, 256)
        c4 = self.pool(c3_2)  # (14, 14, 256) -> (7, 7, 256)
        c5 = F.relu(self.fc1(c4.view(c4.shape[0], -1)))  # (7, 7, 256) -> 512
        c6 = self.fc2(c5)  # 512 -> 7
        c7 = F.softmax(c6, dim=1)

        return c7


class SubNet2(nn.Module):

    """ Backbone's sub-network2 for other scale features  """

    def __init__(self):
        super(SubNet2, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.fc1 = nn.Linear(7 * 7 * 128, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        c1_1 = F.relu(self.conv1(self.pool(x)))  # (56, 56, 64) -> (28, 28, 128)
        c1_2 = F.relu(self.conv1_2(c1_1))  # (28, 28, 128) -> (28, 28, 128)
        c2_1 = F.relu(self.conv2(self.pool(c1_2)))  # (28, 28, 128) -> (14, 14, 128)
        c2_2 = F.relu(self.conv2(c2_1))  # (14, 14, 128) -> (14, 14, 128)
        c3 = self.pool(c2_2)  # (14, 14, 128) -> (7, 7, 128)
        c4 = F.relu(self.fc1(c3.view(c3.shape[0], -1)))  # (7, 7, 128) -> 512
        c5 = self.fc2(c4)  # 512 -> 7
        c6 = F.softmax(c5, dim=1)

        return c6


class CombNet(nn.Module):

    """ Merge FeaturePyramidStructure, SubNet1, SubNet2 """

    def __init__(self, device):
        super(CombNet, self).__init__()
        self.FPN = FeaturePyramidStructure()
        self.sub1 = SubNet1()
        self.sub2 = SubNet2()

        self.FPN.to(device)
        self.sub1.to(device)
        self.sub2.to(device)

    def forward(self, x):
        topdown_1, f2, f3 = self.FPN(x)
        topdown_2 = self.sub2(f2)
        topdown_3 = self.sub1(f3)

        return topdown_1, topdown_2, topdown_3


device = torch.device(3 if torch.cuda.is_available() else torch.device('cpu'))
combNet = CombNet(device)
combNet.to(device)

# For comparison with plain resnet-18
resnet18 = FeaturePyramidStructure()
resnet18.to(device)

combNet.train()
resnet18.train()

params = [p for p in combNet.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

params_res = [p for p in resnet18.parameters() if p.requires_grad]
optimizer_res = torch.optim.SGD(params_res, lr=0.01, momentum=0.9)
criterion_res = nn.CrossEntropyLoss()

# best
alpha = 0.5
beta = 0.7

# alpha = 0.1
# beta = 0.2

# For plotting
comb_log = []
res_log = []

for epoch in range(26):
    for iter, (img, target) in enumerate(train_loader):
        img = img.float().to(device)
        img = img / 255
        target = target.to(device)

        optimizer.zero_grad()
        optimizer_res.zero_grad()

        loss1 = criterion(combNet(img)[0], torch.argmax(target, dim=1))
        loss2 = criterion(combNet(img)[1], torch.argmax(target, dim=1))
        loss3 = criterion(combNet(img)[2], torch.argmax(target, dim=1))

        loss_res = criterion_res(resnet18(img)[0], torch.argmax(target, dim=1))

        tloss = (loss1 + alpha * loss2 + beta * loss3)
        tloss_res = loss_res

        with torch.no_grad():
            acc = torch.true_divide(torch.sum(torch.argmax(combNet(img)[0], dim=1) == torch.argmax(target, dim=1)),
                                    len(target)) * 100
            print('{}/{} combNet loss:{}, acc : {}'.format(epoch, iter, tloss, acc))
            comb_log.append(acc)

            acc_res = torch.true_divide(torch.sum(torch.argmax(resnet18(img)[0], dim=1) == torch.argmax(target, dim=1)),
                                        len(target)) * 100
            print('     resnet18 loss:{}, acc : {}'.format(tloss_res, acc_res))
            res_log.append(acc_res)

        tloss.backward()
        tloss_res.backward()
        optimizer.step()
        optimizer_res.step()

    plt.plot(np.arange(len(comb_log)), comb_log, label="combnet acc")
    plt.plot(np.arange(len(res_log)), res_log, label="resnet18 acc")
    plt.legend(loc='best')
    plt.show()