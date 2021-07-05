from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from kinship_dataset import KinDataset
from kinship_model_basic import SiameseNet
from kinship_predict import KinDatasetTest
from kinship_utils import free_gpu_cache

print("Prepare data...")
train_file_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train_list.csv"
train_folders_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train/"
val_famillies = ["family10", "family4"]

all_images = glob(train_folders_path + '/*/*/*.jpg')
all_images = [x.replace("\\", "/") for x in all_images]
all_files = [str(i).split("/")[-1][:-4] for i in all_images]

# Filter step fOr bounding boxes
delete_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\bounding boxes alligment" \
              "\\delete list.txt"
delete_file = pd.read_csv(delete_path, delimiter=";")
FILTERS = "maj_oob,mnr_oob,blr,ilu,drk,grn,lbl"
filters = FILTERS.replace(" ", "")
filters = filters.split(",")
deleted = []
for i in all_files:
    check = delete_file["filename"] == i
    check = list(np.where(check)[0])
    if len(check) != 0:
        d_fs = delete_file["filter"].iloc[check]
        current_f = list(d_fs.values)[0].split(',')
        for f in current_f:
            if f in filters:
                to_delete = all_files.index(i)
                deleted.append(to_delete)
                break

# delete filtered from all images
all_images = [all_images[i] for i in range(len(all_images)) if i not in deleted]

train_images = []
val_images = []
for x in all_images:
    for i in range(len(val_famillies)):
        if val_famillies[i] not in x:
            train_images.append(x)
        elif val_famillies[i] in x:
            val_images.append(x)

train_person_to_images_map = defaultdict(list)
ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]  # family/person
for x in train_images:
    # append train image to the person in the above format
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

# similar but validation
val_person_to_images_map = defaultdict(list)
for x in val_images:
    # same as above but val images
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

# get pairs from csv to a zipped list
relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
# validate for people
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

# get train and val relationship list
train_relations = []
val_relations = []

for i in range(len(val_famillies)):
    for x in relationships:
        if val_famillies[i] not in x[0]:
            train_relations.append(x)
        elif val_famillies[i] in x[0]:
            val_relations.append(x)


# Prepare data loaders
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45, expand=True),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.Resize(224),
    transforms.ToTensor(),  # this transforms values to [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])  # this transforms values to [-1,1]

])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

test_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\test\\"
test_file = 'D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\test.csv'

train_set = KinDataset(train_relations, train_person_to_images_map, train_transform)
val_set = KinDataset(val_relations, val_person_to_images_map, val_transform)
test_set = KinDatasetTest(test_path, test_file, test_transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# network and parameters
print("Initialize network...")
free_gpu_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL
net = SiameseNet().to(device)

lr = 1e-3  # learning rate

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, patience=10)


def train():
    net.train()
    train_loss = 0.0
    running_loss = 0.0
    running_corrects = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1, 1).to(device)
        output = net(img1, img2)
        predictions = output > 0.5

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        running_loss += loss.item()
        running_corrects += torch.sum(predictions == (label > 0.5))

        step = 100
        if i % step == step - 1:
            print(' [{} - {:.2f}%],\ttrain loss: {:.5}'.format(epoch + 1, 100 * (i + 1) / len(train_loader),
                                                               running_loss / step / 200))
            running_loss = 0

    train_loss /= len(train_set)
    running_corrects = running_corrects.item() / len(train_set)
    print('[{}], \ttrain loss: {:.5}\tacc: {:.5}'.format(epoch + 1, train_loss, running_corrects))
    return train_loss, running_corrects


def validate():
    net.eval()
    val_loss = 0.0
    running_corrects = 0

    for batch in val_loader:
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1, 1).to(device)
        with torch.no_grad():
            output = net(img1, img2)
            predictions = output > 0.5
            loss = criterion(output, label)

        val_loss += loss.item()
        running_corrects += torch.sum(predictions == (label > 0.5))

    val_loss /= len(val_set)
    running_corrects = running_corrects.item() / len(val_set)
    print('[{}], \tval loss: {:.5}\tacc: {:.5}'.format(epoch + 1, val_loss, running_corrects))

    return val_loss, running_corrects


def test(net, test_loader):
    predictions = []
    batch_num = 1
    for batch in test_loader:
        img1, img2, label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.float().view(-1, 1).to(device)
        with torch.no_grad():
            output = net(img1, img2)
            values = output.tolist()
            for v in values:
                predictions.append(v[0])
        print("test batch " + str(batch_num))
        batch_num += 1
    return predictions


print("Start training...")
# main training parameters
num_epoch = 1

best_val_loss = 1000
best_epoch = 0

history = []
accuracy = []
for epoch in range(num_epoch):
    train_loss, train_acc = train()
    val_loss, val_acc = validate()
    history.append((train_loss, val_loss))
    accuracy.append((train_acc, val_acc))
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.torch.save(net.state_dict(), 'net_checkpoint.pth')

torch.save(net.state_dict(), 'net_full_training.pth')

plt.plot([x[0] for x in history], 'b', label='train')
plt.plot([x[1] for x in history], 'r--', label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


predictions = test(net, test_loader)
results = pd.read_csv(test_file)

results['is_related'] = predictions
results.to_csv("kinship_results_basic.csv", index=False)
print("done")
