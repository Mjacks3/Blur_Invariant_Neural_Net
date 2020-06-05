from resnet_models import resnet152
#import layers
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import distance
import numpy as np


cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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
criterion = nn.MSELoss(reduce=True, reduction='sum')
#criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(resnet.parameters(), lr= 1e-4)


for epoch in range(1): 

    batch_size = 4
    for batch_init in range(0,8,batch_size):

        batch_end = batch_init + batch_size
        inputs_flat = []
        labels = []
        outputs = []

        #For each Batch, Load images and apply transformations
        for file_name in train_pairs[batch_init:batch_end]:
            file_pieces = file_name.split()

            if len(file_pieces) == 5: # Matching Pair 
                labels.append(1) #1 is true 

                pair_a_file_name = ("data/lfw_blur_"+ file_pieces[2] +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
                im_a = Image.open(pair_a_file_name)
                tensor_a  =  transform(im_a)

                pair_b_file_name = ("data/lfw_blur_"+ file_pieces[4] +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[3])) +file_pieces[3]+".jpg"  )
                im_b = Image.open(pair_b_file_name)
                tensor_b  =  transform(im_b)

                inputs_flat.append(tensor_a)
                inputs_flat.append(tensor_b)


            elif len(file_pieces) == 6: # Non-Matching Pair 
                labels.append(0) #0 is false 


                pair_a_file_name = ("data/lfw_blur_"+ file_pieces[2] +"/"+file_pieces[0]+"/"+file_pieces[0]+ "_"+  "0"*(4 - len(file_pieces[1])) +file_pieces[1]+".jpg"  )
                im_a = Image.open(pair_a_file_name)
                tensor_a  =  transform(im_a)

                pair_b_file_name = ("data/lfw_blur_"+ file_pieces[5] +"/"+file_pieces[3]+"/"+file_pieces[3]+ "_"+  "0"*(4 - len(file_pieces[4])) +file_pieces[4]+".jpg"  )
                im_b = Image.open(pair_b_file_name)
                tensor_b  =  transform(im_b)

                inputs_flat.append(tensor_a)
                inputs_flat.append(tensor_b)


            else: 
                print("error")
                print(line) 
                exit()
        

        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize


        stacked_inputs = torch.stack(inputs_flat) # convert the list of tensors to a single tensor


        intermediates = resnet(stacked_inputs) #Feature Extraction?


        
        features = intermediates.data # get the tensor out of the variable

        
        #Take extracted tensore pairs and compare them to find Binary classification predicition
        for pair_tensor in range(0,batch_size*2,2):
            d = distance.euclidean(features[pair_tensor], features[pair_tensor+1])
            
            print(d)

            #cosine similarity
            #cos_sim = cos(features[pair_tensor].squeeze(0),features[pair_tensor+1].squeeze(0))
            #print(cos_sim)

            if d < 5: #unsimilar
                outputs.append(1)
            else: 
                outputs.append(0)


        
        #Loss based between binary classification
        loss = criterion(np.array(outputs), np.array(labels))
        loss.backward()
        optimizer.step()



    """
#PATH = "saved_models/resnet_152"
#torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()
    """

