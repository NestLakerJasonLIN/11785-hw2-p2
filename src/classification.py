
# coding: utf-8

# In[5]:


import time
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import glob
from PIL import Image

verbose = True
mode = "development"
cuda = torch.cuda.is_available()
num_workers = 4 if cuda else 0 
device = torch.device("cuda" if cuda else "cpu")

root = "../data/"
pred = "../pred/"

pred_cls_filename = pred + "test_cls_pred.csv"
pred_vrf_filename = pred + "test_vrf_pred.csv"
dataset_cat = "medium"

eval_cls = root + "validation_classification/" + dataset_cat
test_cls = root + "test_classification/" + "medium"
if (mode=="development"):
    train_cls = eval_cls # for development
else:
    train_cls = root + "train_data/" + dataset_cat # for actual training

eval_vrf = root + "validation_verification"
test_vrf = root + "test_verification"

test_cls_order_path = root + "test_order_classification.txt"
test_vrf_order_path = root + "test_trials_verification_student.txt"

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# hyper-parameters
input_shape = torch.Size([3, 32, 32])
epochs = 100
batch_size = 256
dropout = 0.2
num_faceids = 2300

if verbose:
    print("mode: %s" % mode)
    print("torch version: %s" % torch.__version__)
    print("np version: %s" % np.__version__)
    print("cuda: %s" % cuda)
    print("num_workers: %s" % num_workers)
    print("device: %s" % device)
    print("verbose: %s" % verbose)



# In[8]:

class testClassfiyDataset(Dataset):
    def __init__(self, test_path, transforms, test_cls_order_path):
        super().__init__()

        self.test_path = test_path
        self.transforms = transforms
        
        # load image order file
        self.image_order_list = np.loadtxt(test_cls_order_path, dtype=str)

    def __len__(self):
        return len(self.image_order_list)
      
    def __getitem__(self, index):
        image_rel_path = self.image_order_list[index]
        image_path = self.test_path + "/" + image_rel_path
        image = Image.open(image_path)
        image = self.transforms(image)
        
        return image

# In[11]:


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()   # .backward() accumulates gradients

        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        predicted.detach_()
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    end_time = time.time()

    running_loss /= len(train_loader)
    acc = (correct_predictions / total_predictions) * 100.0
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc, '%')
    return running_loss

def evaluate_model(model, eval_loader, criterion, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()

        running_loss /= len(eval_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('evaluate Loss: ', running_loss)
        print('evaluate Accuracy: ', acc, '%')
        return running_loss, acc

def test_model(model, test_loader, device, save=False, filename="../data/test_pred.csv"):
    predicts = torch.LongTensor().to(device)
    
    with torch.no_grad():
        model.eval()

        model.to(device)

        # no target in test dataset/data loader
        for batch_idx, data in enumerate(tqdm(test_loader)):
            data = data.to(device)

            outputs = model(data)

            _, predict = torch.max(outputs.data, 1)
            
            predicts = torch.cat([predicts, predict])
    
    assert predicts.shape[0] == len(test_loader.dataset)
    assert predicts.shape[0] == len(test_loader.dataset.image_order_list)
    
    if save:
        # convert label index back to real indentity label
        predict_labels = []
        for i in predicts.detach().cpu().numpy():
            predict_labels.append(
                [key  for (key, value) in train_dataset.class_to_idx.items() if value == i][0])
        
        result = np.concatenate([test_loader.dataset.image_order_list.reshape(-1, 1),
                                 np.asarray(predict_labels).reshape(-1, 1)], axis=1)
        np.savetxt(filename, result, fmt="%s", delimiter=",", header="Id,Category", comments="")
    
    return predicts

def train_model(model, epochs, train_loader, eval_loader, criterion, optimizer, device, scheduler=None):
    model.to(device)

    for epoch in range(epochs):
        print("epoch: %d" % (epoch))
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device=device)
        eval_loss, eval_acc = evaluate_model(model, eval_loader, criterion, device=device)
        
        if scheduler:
            scheduler.step(eval_loss)
        
        print('=' * 20)
    
    return 


# In[12]:


# reference: https://pytorch.org/docs/stable/_modules/torchvision/models/mobilenet.html
class Conv2dBNReLU6(nn.Module):
    # per paper:  All spatial convolutions use 3 × 3 kernels
    def __init__(self, cin, cout, ks=3, sd=1, pd=-1, grp=1, relu=True):
        super(Conv2dBNReLU6, self).__init__()
        
        if (pd<0):
            # reference: http://cs231n.github.io/convolutional-networks/
            pd = (ks - 1) // 2
        
        layers = []
        
        # convolution layer, TODO: bias=False?
        layers.append(nn.Conv2d(cin, cout, ks, sd, pd, groups=grp, bias=False))
        
        # batch norm layer
        layers.append(nn.BatchNorm2d(cout))
        
        # relu layer
        if (relu):
            # use inplace to improve memory usage
            layers.append(nn.ReLU6(inplace=True))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class BottleNeck(nn.Module):
    def __init__(self, cin, t, cout, sd):
        super(BottleNeck, self).__init__()
        
        self.cin = cin
        self.cout = cout
        self.sd = sd
        
        layers = []
        
        # expanision : 1x1 conv2d , ReLU6
        c_expan = cin * t
        
        # only expan if expansion ratio is larger than 1
        if (t != 1):
            layers.append(Conv2dBNReLU6(cin, c_expan, ks=1, sd=1))
        
        # depth-wise convolution : 3x3 dwise s=s, ReLU6
        layers.append(Conv2dBNReLU6(c_expan, c_expan, ks=3, sd=sd, grp=c_expan))
        
        # projection : linear 1x1 conv2d, no ReLU6
        layers.append(Conv2dBNReLU6(c_expan, cout, ks=1, sd=1, pd=0, relu=False))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.net(x)
        # use residual if input/output has same shape
        if (self.cin == self.cout and self.sd == 1):
            out += x
        return out
        
# reference: https://pytorch.org/docs/stable/_modules/torchvision/models/mobilenet.html
class MobileNetV2(nn.Module):
    def __init__(self, in_shape, output_size, dropout=0.2):
        super(MobileNetV2, self).__init__()

        # all hyper-parameters
        cin, w, h = in_shape[0], in_shape[1], in_shape[2]
        cout = 32 # first layer's output channels
        # TUNE: stride
        # t, c, n, s
        bottleneck_architects = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        c_last = 4096 # TUNE: last channel from bottlenecks

        layers = []

        # 1. first conv2d layer : kernel_size = 3 stride = 1, cin, cout, ks, sd
        layers.append(Conv2dBNReLU6(cin, cout, 3, 1))
        cin = cout

        # 2. 17 bottleneck blocks
        for t, c, n, s in bottleneck_architects:
            cout = c
            for i in range(n):
                # per paper: The first layer of each sequence has a stride s and all others use stride 1.
                s = s if i == 0 else 1
                # cin, t, cout, sd                
                layers.append(BottleNeck(cin, t, cout, s))
                cin = cout

        # 3. last conv2d layer: cout=c_last, ks = 1 sd = 1
        layers.append(Conv2dBNReLU6(cin, c_last, 1, 1))

        self.feature_extractor = nn.Sequential(*layers); # separate for verification task
        
        # 4. avgpool layer avgpool convert to only 1 feature
        # trick: use torch.mean to finish this in forward method later

        # 5. classify layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_last, output_size)
            )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # get embedding feature
        x = self.feature_extractor(x)
        
        # avgpool layer
        x = x.mean([2, 3])

        # classify layer
        x = self.classifier(x) 

        return x

if __name__ == "__main__":
    # load dataset
    if (verbose):
        print("loading dataset...")

    train_dataset = datasets.ImageFolder(root = train_cls, transform=transformations)
    eval_dataset = datasets.ImageFolder(root = eval_cls, transform=transformations)
    test_dataset = testClassfiyDataset(test_cls, transformations, test_cls_order_path)

    if (verbose):
        print("load train dataset: ", len(train_dataset))
        print("load eval dataset: ", len(eval_dataset))
        print("load test dataset: ", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,              # The dataset
        batch_size=batch_size,      # Batch size
        shuffle=True,               # Shuffles the dataset at every epoch
        pin_memory=True,            # Copy data to CUDA pinned memory
        num_workers=num_workers     # Number of worker processes for loading data.
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    # define model
    if verbose:
        print("define model...")
    model = MobileNetV2(in_shape=input_shape,
                        output_size=num_faceids, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.to(device).parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # training
    if verbose:
        print("training...")

    train_model(model, epochs, train_loader, eval_loader, criterion, optimizer, device, scheduler)

    # predicting
    if verbose:
        print("predicting...")

    predicts = test_model(model, test_loader, device, save=True, filename=pred_cls_filename)

    if verbose:
        print("finished")