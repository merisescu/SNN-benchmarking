from tonic import transforms
import numpy as np
#import time

#%% NMNIST

class MNIST_loader:
    def __init__(self):
        self.group_size = 1
        self.train_dataset = None
        self.train_idx = None
        self.val_idx = None
        self.test_dataset = None
        self.test_idx = None
        self.active_pixels = None
        self.current_train_idx = 0
        self.current_val_idx = 0
        self.current_test_idx = 0
        self.t_max = 336040

    def setup_MNIST(self,activity_threshold=0.02,val_split=0.1,show_plots=False):
        from tonic.datasets import NMNIST
        self.train_dataset = NMNIST(save_to="./", train=True)
        self.test_dataset = NMNIST(save_to="./", train=False)

        self.test_idx = np.arange(len(self.test_dataset))
        self.train_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(self.test_idx)
        np.random.shuffle(self.train_idx)

        val_split_idx = int((1-val_split) * len(self.train_dataset))
        self.val_idx = self.train_idx[val_split_idx:]
        self.train_idx = self.train_idx[:val_split_idx]

        if show_plots:
            import matplotlib.pyplot as plt
        try:
            self.active_pixels = np.loadtxt("pixel_activations.csv", skiprows=1, delimiter=",",dtype=[('p',np.int8),('x',np.int8),('y',np.int8)])
        except:
            sensor_size = NMNIST.sensor_size
            frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)

            data, labels = self.train_dataset[0]
            average = frame_transform(data).astype(np.float32)

            # collect pixel activations per trial and average
            labelcounter = 0
            for i in self.train_idx-1:
                data, labels = self.train_dataset[i+1]
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
            self.active_pixels = np.loadtxt("pixel_activations.csv", skiprows=1, delimiter=",",dtype=[('p',np.int8),('x',np.int8),('y',np.int8)])
        self.active_pixels = self.active_pixels[['x','y','p']]
        self.n_in = self.active_pixels.size
        num_classes = 10

        return self.n_in, num_classes

    '''def base_loader(self,dataset,index_list,current_index):
        end_index = current_index + self.group_size
        
        selected_indices = np.take(index_list, range(current_index, end_index), mode="wrap")

        current_index = (current_index + self.group_size) % len(index_list)

        spikes = [[] for _ in selected_indices]
        labels = []
        for item, idx in enumerate(selected_indices):
            data, label = dataset[idx]
            labels.extend([label])
            
            for px in range(self.n_in):
                mask = np.isin(data[['x','y','p']],self.active_pixels[px])
                relative_times = data[mask]["t"]
                spikes[item].extend([relative_times])
        
        return spikes,labels

    def training_loader(self):
        return self.base_loader(self.train_dataset,self.train_idx,self.current_train_idx)

    def validation_loader(self):
        return self.base_loader(self.train_dataset,self.val_idx,self.current_val_idx)

    def test_loader(self):
        return self.base_loader(self.test_dataset,self.test_idx,self.current_test_idx)'''

    def training_loader(self):
        end_index = self.current_train_idx + self.group_size

        selected_indices = np.take(self.train_idx, range(self.current_train_idx, end_index), mode="wrap")

        self.current_train_idx = (self.current_train_idx + self.group_size) % len(self.train_idx)

        spikes = [[] for _ in selected_indices]
        labels = []
        for item, idx in enumerate(selected_indices):

            data, label = self.train_dataset[idx]
            labels.extend([label])
            
            for px in range(self.n_in):
                mask = np.isin(data[['x','y','p']],self.active_pixels[px])
                relative_times = data[mask]["t"]
                spikes[item].extend([relative_times])

        return spikes,labels

    def validation_loader(self):
        end_index = self.current_val_idx + self.group_size
        
        selected_indices = np.take(self.val_idx, range(self.current_val_idx, end_index), mode="wrap")

        self.current_val_idx = (self.current_val_idx + self.group_size) % len(self.val_idx)

        spikes = [[] for _ in selected_indices]
        labels = []
        for item, idx in enumerate(selected_indices):
            
            data, label = self.train_dataset[idx]
            labels.extend([label])
            
            for px in range(self.n_in):
                mask = np.isin(data[['x','y','p']],self.active_pixels[px])
                relative_times = data[mask]["t"]
                spikes[item].extend([relative_times])

        return spikes,labels

    def test_loader(self):
        end_index = self.current_test_idx + self.group_size
        
        selected_indices = np.take(self.test_idx, range(self.current_test_idx, end_index), mode="wrap")

        self.current_test_idx = (self.current_test_idx + self.group_size) % len(self.test_idx)

        spikes = [[] for _ in selected_indices]
        labels = []
        for item, idx in enumerate(selected_indices):

            data, label = self.test_dataset[idx]
            labels.extend([label])
            
            for px in range(self.n_in):
                mask = np.isin(data[['x','y','p']],self.active_pixels[px])
                relative_times = data[mask]["t"]
                spikes[item].extend([relative_times])

        return spikes,labels

#%% Spiking speech commands
import tonic.datasets
import h5py
import os
from tonic.io import make_structured_array
class SSC(tonic.datasets.SSC):
    def __getitem__(self, index):
        file = h5py.File(
            os.path.join(self.location_on_system, self.data_filename), "r"
        )

        events = make_structured_array(
            file["spikes/times"][index].astype(np.float32) * 1e6,
            file["spikes/units"][index],
            1,
            dtype=self.dtype,
        )

        target = file["labels"][index].astype(int)

        if self.transform is not None:
            events = self.transform(events)

        if self.target_transform is not None:
            target = self.target_transform(target)

        file.close()
        return events, target

class SSC_loader:
    def __init__(self):
        self.group_size = 1
        self.train_dataset = None
        self.train_idx = None
        self.val_dataset = None
        self.val_idx = None
        self.test_dataset = None
        self.test_idx = None
        self.active_pixels = None
        self.current_train_idx = 0
        self.current_val_idx = 0
        self.current_test_idx = 0
        self.all_keywords = True
        self.t_max = 1e6

    def setup_speech_commands(self,all_keywords=True,activity_threshold=0.02):
        if all_keywords:
            self.num_classes=35
        else:
            self.num_classes=11
        self.n_in = 700
        self.train_dataset = SSC(save_to='./',split='train')
        self.val_dataset = SSC(save_to='./',split='valid')
        self.test_dataset = SSC(save_to='./',split='test')
        self.train_idx = np.arange(len(self.train_dataset))
        self.val_idx = np.arange(len(self.val_dataset))
        self.test_idx = np.arange(len(self.test_dataset))
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.val_idx)
        np.random.shuffle(self.test_idx)
        self.active_pixels = np.loadtxt("SSC_channel_activations.csv", skiprows=1, delimiter=",",dtype=[('x',np.int16)])
        try:
            self.active_pixels = np.loadtxt("SSC_channel_activations.csv", skiprows=1, delimiter=",",dtype=[('x',np.int16)])
        except:
            sensor_size = SSC.sensor_size
            frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)

            data, labels = self.train_dataset[0]
            average = frame_transform(data).astype(np.float32)

            # collect pixel activations per trial and average
            labelcounter = 0
            for i in self.train_idx-1:
                data, labels = self.train_dataset[i+1]
                frame = frame_transform(data).astype(np.float32)
                average = np.add(average, frame)
            average = average.squeeze()
            average = np.divide(average, np.max(average))

            active_pixels = np.argwhere(average > activity_threshold) # get indices of active pixels, i think doing it this way is faster and more natural than a binary mask considering the event based nature

            np.savetxt("SSC_channel_activations.csv", X=active_pixels, header="x", delimiter=",",fmt='%i')
            self.active_pixels = np.loadtxt("SSC_channel_activations.csv", skiprows=1, delimiter=",",dtype=[('x',np.int16)])
        self.n_in = self.active_pixels.size
    
        return self.n_in, self.num_classes

    def training_loader(self):
        end_index = self.current_train_idx + self.group_size
        
        selected_indices = np.take(self.train_idx, range(self.current_train_idx, end_index), mode="wrap")

        self.current_train_idx = (self.current_train_idx + self.group_size) % len(self.train_idx)

        spikes = [[] for _ in selected_indices]
        labels = []

        for item, idx in enumerate(selected_indices):
        
            data, label = self.train_dataset[idx]
            labels.extend([np.min([label,self.num_classes])])
            
            for px in range(self.n_in):
                mask = np.isin(data[['x']],self.active_pixels[px])
                relative_times = data[mask]["t"]
                spikes[item].extend([relative_times])

        return spikes, labels
    
    def validation_loader(self):
        end_index = self.current_val_idx + self.group_size
                
        selected_indices = np.take(self.val_idx, range(self.current_val_idx, end_index), mode="wrap")

        self.current_val_idx = (self.current_val_idx + self.group_size) % len(self.val_idx)

        spikes = [[] for _ in selected_indices]
        labels = []

        for item, idx in enumerate(selected_indices):
                
                    data, label = self.val_dataset[idx]
                    labels.extend([label])
                    
                    for px in range(self.n_in):
                        mask = np.isin(data[['x']],self.active_pixels[px])
                        relative_times = data[mask]["t"]
                        spikes[item].extend([relative_times])
        
        return spikes, labels

    def test_loader(self):
        end_index = self.current_test_idx + self.group_size
                
        selected_indices = np.take(self.test_idx, range(self.current_test_idx, end_index), mode="wrap")

        self.current_test_idx = (self.current_test_idx + self.group_size) % len(self.test_idx)

        spikes = [[] for _ in selected_indices]
        labels = []

        for item, idx in enumerate(selected_indices):
                
                    data, label = self.test_dataset[idx]
                    labels.extend([label])
                    
                    for px in range(self.n_in):
                        mask = np.isin(data[['x']],self.active_pixels[px])
                        relative_times = data[mask]["t"]
                        spikes[item].extend([relative_times])

        return spikes, labels