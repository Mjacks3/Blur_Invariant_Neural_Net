from resnet_models import resnet152
#import layers
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image



#Resize to the 224 input range

transform = transforms.Compose(
    [#transforms.Scale((224,224)),  # resized to the network's required input size
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



#Read in pairs:
train_pairs = []
with open("data/pairsDevTrainWithBlur.txt", 'r') as fi:  
    for line in fi:
        if len(line.split()) >= 3:
            train_pairs.append(line)


batch_size = 4
for batch in train_pairs[0:8:batch_size]:
    
    print(batch[0:batch_size])




# Read in and  load data (train and test) 
#read pairs and make list with origal dataset first if you want then blurred

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





resnet = resnet152()
#criterion = nn.MSELoss(reduce=True, reduction='sum')
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(resnet.parameters(), lr= 1e-4)

########################################################################
# Train the network
########################################################################

for epoch in range(2):  # loop over the dataset multiple times
    print("NEW Epoch")

    running_loss = 0.0
    print(len(trainloader))
    

    for i, data in enumerate(trainloader, 0):
        print("enumerate")
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(len(labels))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        print(type(inputs))
        print(len(inputs))
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

"""
print('Finished Training')



#PATH = "saved_models/resnet_152"
#torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()
"""

