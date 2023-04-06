import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from cv2 import imread


class SpectrogramDataset(Dataset):
    # custom dataset to load spectrogram data
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = [self.labels.iloc[i, 1]
                        for i in range(len(self.labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        image = imread(img_path)    # io.imread reads 4 channels instead of 3
        y_label = torch.tensor(int(self.labels.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

    def getItemLabel(self, index):
        """
            Takes the true label of the item at the given index
            
            Returns
            ----
            label : str
                the true label represented by one of the possible strings: ("FLAC", "V0", "320K", "192K", "128K")
        """
        return self.labels.iloc[index, 1]


def get_all_preds(net, loader, classes, printAcc=True):
    """
        Runs the batch of data with the neural network provided. 
        Returns the predictions and prints accuracy results if specified.

        Parameters
        ----
        net : nn.Module()
            neural network architecture to use to predict

        loader : torch.utils.data.DataLoader
        
        classes : set
            set of possible classification targets
        
        printAcc : bool
            flag used to determine if printing accuracy is necessary
        
        Returns
        ----
        all_preds : torch.tensor
            tensor containing predictions of the given data
    """
    all_preds = torch.tensor([])
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    for batch in loader:
        images, labels = batch

        preds = net(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )

        # record total number of predictions and number of correct predictions
        for label, predictions in zip(labels, torch.max(preds, 1)[1]):
            if label == predictions:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    if printAcc:
        for classname, count in correct_pred.items():
            accuracy = 100 * float(count) / total_pred[classname]
            print(
                f"Accuracy {classname:4s}: {count} / {total_pred[classname]} = {accuracy:.2f}%")

    return all_preds


def displayProgress(progress, total, loss=-1, rloss=-1, scale=30):
    """
        Displays a progress bar depending on the data given.

        Parameters
        ----
        progress : int

        total : int

        loss : float

        rloss : float

        scale : int > 0
    """
    current = int(scale * (progress / float(total)))
    bar = "█" * current + "-" * (scale - current)

    if loss == -1 and loss == rloss:
        print(
            f"\r  │{bar}│ Progress {progress:3d}/{total:<3d} │", end="\r")
    else:
        print(
            f"\r  │{bar}│ Step {progress:3d}/{total:<3d} │ Loss: {loss:.5f} RLoss: {rloss:.5f}", end="\r")


def dictCounter(array):
    """
        Utility function to count the frequency of data inside an array/list

        Returns
        ----
        counter : dict
            contains the frequency count of each element from the array/list
    """
    counter = {}
    for i in array:
        counter[i] = counter.get(i, 0) + 1

    return dict(sorted(counter.items()))


def imshow(img):
    """
        Display image using matplotlib
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def viewRandomSample(data_loader, classes, batch_size):
    """
        Take random images from dataloader and display them
    """
    # get some random training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # show images & labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))


def stratifiedSampling(dataset, k):
    """
       Takes k samples of each classification from the dataset.
       Current broken.
    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for data, label in dataset:
        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1

        if class_counts[c] <= k:
            train_data.append(data)
            train_label.append(torch.unsqueeze(label, 0))
        else:
            test_data.append(data)
            test_label.append(torch.unsqueeze(label, 0))

    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label)
    test_data = torch.cat(test_data)
    test_label = torch.cat(test_label)

    return (torch.utils.data.TensorDataset(train_data, train_label),
            torch.utils.data.TensorDataset(test_data, test_label))
