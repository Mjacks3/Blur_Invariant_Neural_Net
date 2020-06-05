from resnet_models import resnet152
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
        
        print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        #TODO  Means Squared error 
        #TODO: Implement loss function that adds a layer to calculate loss 
        #TODO: Reduce mean error in tensor 
        #TODO TAKE YHAT AND TRUE VALUE  AND DOES SUBTRACTION AND MEAN SQUARE
        #TODO PASS THROUGH 
        #IF you can't take a layer and do output.backward() from that layer outputting a single number, then you can just use l1 losss
        #MIX THEM ALL TOGETHER.
        #Get any model working based 

        

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        break

print('Finished Training')


#PATH = "saved_models/resnet_152"
#torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()


correct = 0
total = 0
with torch.no_grad():

    for data in testloader:
        images, labels = data
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

