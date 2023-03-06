import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from cv2 import imread


class SpectrogramDataset(Dataset):
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
        return self.labels.iloc[index, 1]


def get_all_preds(net, loader, classes, printAcc=False):
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


def dictCounter(array):
    counter = {}
    for i in array:
        counter[i] = counter.get(i, 0) + 1

    return dict(sorted(counter.items()))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def viewRandomSample(data_loader, classes, batch_size):
    # get some random training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # show images & labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))

# broken rn


def stratifiedSampling(dataset, k):
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
