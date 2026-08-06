from tonic import transforms
import numpy as np

#%% NMNIST

class MNIST_loader:
    def __init__(self):
        self.train_dataset = None
        self.train_idx = None
        self.val_idx = None
        self.test_dataset = None
        self.test_idx = None
        self.active_pixels = None
        self.train_batch = 0
        self.val_batch = 0
        self.test_batch = 0

    def setup_MNIST(self,activity_threshold=0.02,val_split=0.1,show_plots=False):

        self.train_dataset = NMNIST(save_to="./", train=True)
        self.test_dataset = NMNIST(save_to="./", train=False)

        self.test_idx = np.arange(len(self.test_dataset))
        self.train_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(self.test_idx)
        np.random.shuffle(self.train_idx) # randomise data order

        val_split_idx = int((1-val_split) * len(self.train_dataset))
        self.val_idx = self.train_idxs[val_split_idx:]
        self.train_idx = self.train_idx[:val_split_idx]

        if show_plots:
            import matplotlib.pyplot as plt
        from tonic.datasets import NMNIST
        try:
            self.active_pixels = np.loadtxt("pixel_activations.csv", skiprows=1, delimiter=",",dtype=[('p',np.int8),('x',np.int8),('y',np.int8)])
        except:
            sensor_size = NMNIST.sensor_size
            frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)

            data, labels = self.train_dataset[0] # initialise shape
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

    def training_loader(self):
        
        return spikes,labels

    def validation_loader(self):
        return spikes,labels

    def test_loader(self):
        return spikes,labels

#%% Spiking speech commands

def setup_speech_commands():
    from tonic.datasets import SSC
    return