import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from aff_resnet import resnet152, resnet18, resnet34, resnet50, resnet101
from kinship_utils import free_gpu_cache

# Find a tutorial on how to train a net on imagenet in pytorch or just copy the basic example and use this loader
# torchvision.datasets.ImageNet(root, split='train')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 4
path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Imagenet\\subset\\data\\"
train_path = path + "train"
test_path = path + "test"

train_set = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


free_gpu_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = resnet152(fuse_type='DAF', small_input=False).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

NUM_EPOCHS = 50


def train():
    net.train()
    running_loss = 0.0
    train_loss = 0.0

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        running_loss += loss.item()

        step = 500
        if i % step == step - 1:
            print(' [{} - {:.2f}%],\ttrain loss: {:.5}'.format(epoch + 1, 100 * (i + 1) / len(train_loader),
                                                               running_loss / step / 200))
            running_loss = 0.0

    train_loss /= len(train_set)
    print('[{}], \ttrain loss: {:.5}'.format(epoch + 1, train_loss))
    return train_loss


history = []
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    train_loss = train()
    history.append(train_loss)

print('Finished Training')

plt.plot([x for x in history], 'b', label='train')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# save model
# PATH = './image_net.pth'
# torch.save(net.state_dict(), PATH)

# TEST for Accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        # images, labels = data  #maybe this?
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
