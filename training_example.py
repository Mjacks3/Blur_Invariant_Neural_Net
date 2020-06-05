from resnet_models import resnet152
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import distance
import numpy as np


#LFW trainig file size is 2200 
#TODO Make sure to use a non remainder batch-size



#Read in lfw data pairs
train_pairs = []
with open("data/pairsDevTrainWithBlur.txt", 'r') as fi:  
    for line in fi:
        if len(line.split()) >= 3:
            train_pairs.append(" " .join(line.split()))


#Prepare Transforms
transform = transforms.Compose(
    [transforms.Resize((224,224)),  # resized to the network's required input size
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Based on ImageNet Mean & STD



#Alternative to torch's built-in dataloader, load the data and train the network
resnet = resnet152()

#optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(resnet.parameters(), lr= 1e-4)



for epoch in range(10): 
    batch_size = 8
    print("Running Epoch " + str(epoch))

    for batch_init in range(0,2200,batch_size):
        print("Processing number " + str(batch_init) + " out of 2200")

        batch_end = batch_init + batch_size

        input_pair_a = []
        input_pair_b = []

        labels = []
        outputs = []

        #For each Batch, Load images and apply transformations
        for file_name in train_pairs[batch_init:batch_end]:
            file_pieces = file_name.split()

            if len(file_pieces) == 5: # Matching Pair 
                labels.append(0) #0 is true 

                pair_a_file_name = ("data/cropped_lfw/lfw_blur_"+ file_pieces[2] +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
                im_a = Image.open(pair_a_file_name)
                tensor_a  =  transform(im_a)

                pair_b_file_name = ("data/cropped_lfw/lfw_blur_"+ file_pieces[4] +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[3])) +file_pieces[3]+".jpg"  )
                im_b = Image.open(pair_b_file_name)
                tensor_b  =  transform(im_b)

                input_pair_a.append(tensor_a)
                input_pair_b.append(tensor_b)




            elif len(file_pieces) == 6: # Non-Matching Pair 
                labels.append(1) #1 is false 


                pair_a_file_name = ("data/cropped_lfw/lfw_blur_"+ file_pieces[2] +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
                im_a = Image.open(pair_a_file_name)
                tensor_a  =  transform(im_a)

                pair_b_file_name = ("data/cropped_lfw/lfw_blur_"+ file_pieces[5] +"/"+file_pieces[3]+"/"+file_pieces[3]+ "_"+  "0"*(4 - len(file_pieces[4])) +file_pieces[4]+".jpg"  )
                im_b = Image.open(pair_b_file_name)
                tensor_b  =  transform(im_b)

                input_pair_a.append(tensor_a)
                input_pair_b.append(tensor_b)



            else: 
                print("Error Processing Line: ")
                print(line) 
                exit()
        

        labels = np.array(labels)
        labels = torch.from_numpy(labels).float() 
        labels = Variable(labels, requires_grad = True)
        

        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        features_pair_a = resnet(torch.stack(input_pair_a)).data # convert the LIST of tensors to a  TENSOR...of tensors
        features_pair_b = resnet(torch.stack(input_pair_b)).data # convert the LIST of tensors to a  TENSOR...of tensors

        euclidean_distance = F.pairwise_distance(features_pair_a, features_pair_b)

        loss_contrastive = torch.mean(  (1 - labels) * torch.pow(euclidean_distance, 2) +
                                      labels * torch.pow(torch.clamp(2 - euclidean_distance, min=0.0), 2))

        print(loss_contrastive)
        loss_contrastive.backward()
        optimizer.step()
        

    torch.save(resnet.state_dict(), 'trained_models/demo'+ str(epoch) +'.pth')
