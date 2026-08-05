# write preprocessed copies of datasets to avoid unneccessary processing during runtime

show_plots = True
activity_threshold = 0.02 # threshold for normalised pixel activity to be considered active

dataset_dir = "./processed/"
NMNIST_dir = "NMNIST"

#%% imports

import os
import tonic
import tonic.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#%% load NMNIST

train_dataset = tonic.datasets.NMNIST(save_to="./", train=True)
test_dataset = tonic.datasets.NMNIST(save_to="./", train=False)

test_idx = np.arange(len(test_dataset))
train_idx = np.arange(len(train_dataset))

#%% get pixel activations

# first reduce dimensions by culling inactive pixels
# this compresses the time dimension of each trial into a single frame

# load active pixels if already calculated saves time
try:
    active_pixels = np.loadtxt("oopixel_activations.csv", skiprows=1, delimiter=",",dtype=[('p',np.int8),('x',np.int8),('y',np.int8)])
except:
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)

    data, labels = train_dataset[0] # initialise shape
    average = frame_transform(data).astype(np.float32)

    # collect pixel activations per trial and average
    labelcounter = 0
    for i in range(len(train_idx)-1):
        data, labels = train_dataset[i+1]
        frame = frame_transform(data).astype(np.float32)
        average = np.add(average, frame)
        if labels != labelcounter:
            print(i,labels)
            labelcounter = labels

    average = average[0]
    average[0] = np.divide(average[0], np.max(average[0]))
    average[1] = np.divide(average[1], np.max(average[1]))
    
    if show_plots:
        fig, axs = plt.subplots(1,2)
        image_on = axs[0].imshow(average[1],cmap='gray')
        image_off = axs[1].imshow(average[0],cmap='gray')
        fig.colorbar(image_on)
        plt.savefig("figures/activation.eps")

    active_pixels = np.argwhere(average > activity_threshold) # get indices of active pixels, i think doing it this way is faster and more natural than a binary mask considering the event based nature

    np.savetxt("pixel_activations.csv", X=active_pixels, header="p, x, y", delimiter=",",fmt='%i')
    active_pixels = np.loadtxt("pixel_activations.csv", skiprows=1, delimiter=",",dtype=[('p',np.int8),('x',np.int8),('y',np.int8)])

#no. samples for each class in the MNIST dataset
MNIST_training=np.asarray([5923,6742,5958,6131,5842,5421,5918,6265,5851,5949])
MNIST_test=np.asarray([980,1135,1032,1010,982,892,958,1028,974,1009])

# TODO: make into structured array when saving not just loading
active_pixels = active_pixels[['x','y','p']]

print('Out of 2312 (34*34*2) pixels: ', active_pixels.size, "are active")
print(len(active_pixels))

n_in = active_pixels.size # number of input neurons (*2 because 2 polarities)

'''#%% pixel hashing

# use a simple hash function to give each pixel a unique hash
def hashpixel(x,y,p,width):
    return y*width*p + x'''

#%% get only active pixels from dataset
import time

def get_active(dataset,index):
    image = []
    data, label = dataset[index]
    for px in range(n_in):
        mask = np.isin(data[['x','y','p']],active_pixels[px])
        relative_times = data[mask]["t"]
        image.append(relative_times)
    print(image)
    return image

labels = range(10)
for label in labels:
    os.makedirs(f"{dataset_dir}{NMNIST_dir}/Train/{label}", exist_ok=True)
    os.makedirs(f"{dataset_dir}{NMNIST_dir}/Test/{label}", exist_ok=True)
    offset_train = 0
    offset_test = 0
    for i in range(MNIST_training[label]):
        start = time.perf_counter()
        np.savez(f"{dataset_dir}{NMNIST_dir}/Train/{label}/{i}.npz",get_active(train_dataset,i+offset_train))
        stop = time.perf_counter()
        print("process time: ",stop-start)
    offset_train = np.sum(MNIST_training[:label])
    for i in range(MNIST_test[label]):
        start = time.perf_counter()
        np.savez(f"{dataset_dir}{NMNIST_dir}/Test/{label}/{i}.npz",get_active(test_dataset,i+offset_test))
        stop = time.perf_counter()
        print("process time: ",stop-start)
    offset_test = np.sum(MNIST_test[:label])
# %%
