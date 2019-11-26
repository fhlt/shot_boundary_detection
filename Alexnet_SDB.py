import imageio
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import cv2 
import os
import torchvision.models as models
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import time

t0 = time.time()
# input video 
filepath = 'test.mp4'
video = imageio.get_reader(filepath)

# extract CNN feature for each frame in video
class pretrainedAlexNet(nn.Module):
    def __init__(self):
        super(pretrainedAlexNet, self).__init__()
        self.my_features = models.alexnet(pretrained = True).features

    def forward(self, input):
        feature_map = self.my_features(input)
        return feature_map

net = pretrainedAlexNet()

num_frame = 0
for i, im in enumerate(video):
    num_frame += 1
    
frame_feature_arr = np.zeros((num_frame, 256, 6, 6))
for i, im in enumerate(video):
    im_np = np.array(im)
    im = Image.fromarray(im_np).resize((224, 224))
    frame = np.array(im)
    
    frame_tensor = torch.from_numpy(frame).transpose(0,1).transpose(0,2).unsqueeze(0).float()
    frame_feature_tensor = net(frame_tensor)
    frame_feature_np = frame_feature_tensor.squeeze(0).data.numpy()
    frame_feature_arr[i] = frame_feature_np

# compare cosine similarity between all consecutive frames
frame_feature_arr = frame_feature_arr.reshape((frame_feature_arr.shape[0], -1))

do_PCA = False

if do_PCA:
    pca = PCA(n_components=100)
    frame_feature_arr_pca = pca.fit(frame_feature_arr).transform(frame_feature_arr)
    cos_sim = np.zeros((frame_feature_arr_pca.shape[0]-1))
    for i in range(frame_feature_arr_pca.shape[0]-1):
        cos_sim[i] = cosine(frame_feature_arr_pca[i+1], frame_feature_arr_pca[i])    
else:
    cos_sim = np.zeros((frame_feature_arr.shape[0]-1))
    for i in range(frame_feature_arr.shape[0]-1):
        cos_sim[i] = cosine(frame_feature_arr[i+1], frame_feature_arr[i])
        
x_index = np.arange(1, frame_feature_arr.shape[0], 1)
'''
plt.figure()
plt.plot(x_index, cos_sim, 'b*')
plt.xlabel('frame in time series')
plt.ylabel('cosine similarity difference')
plt.title('cosine similarity')
'''

boundary_index = np.array([])
for i in range(cos_sim.shape[0]):
    if cos_sim[i] > 0.2:
        boundary_index = np.append(boundary_index, i)


cap = cv2.VideoCapture(filepath) # capture the video from given path
print(len(boundary_index))
count = 0
for frameNum in boundary_index:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frameNum))
    if cap.isOpened():
        ret, frame = cap.read()
        assert ret
        name = os.path.join("./output_shots/","frame_" + str(count).rjust(3, '0')+'.png')
        cv2.imwrite(name, frame)
        count += 1
cap.release()
print(time.time() - t0, 's')
'''
print("start making videos")
# make videos
boundary_index = np.concatenate((np.array([0]),boundary_index))
boundary_index = np.concatenate((boundary_index, np.array([num_frame])))
fps = video.get_meta_data()['fps']

writer_list = []
for bound_ind in range(boundary_index.shape[0]-1):
    print(bound_ind)
    
    writer = imageio.get_writer('output_shots/'+str(bound_ind)+'.gif', fps=fps)
    for i, im in enumerate(video):
        if i > boundary_index[bound_ind]:
            writer.append_data(im)
        if i == boundary_index[bound_ind+1]-1:
            break
    writer.close()
'''
