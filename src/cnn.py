"""
    Script for training and testing a CNN model
    Models are defined in the file `./spectrogram.classifier`
    
    Outputs
    ----
    spectral_net.pt : state_dict
        trained CNN state dictionary
    
    spectral_loss.csv : csv
        contains running loss of each epoch during training
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import spectrogram.classifier as models
import spectrogram.utils as utils

classes = ("FLAC", "V0", "320K", "192K", "128K")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN configurations
num_epochs = 100
batch_size = 32
learning_rate = 0.001
momentum = 0.9

# Paths for dataset and output
data_input = "./data/dataset_3110/spectrograms/"
data_record = "./data/dataset_3110/spectral_record.csv"
net_output = "./spectral_net.pt"
net_record = "./spectral_loss.csv"

display_conf_matrix = False

# change CNN model to use here
net = models.ConvNetKJ().to(device)
net_test = models.ConvNetKJ()


dataset = utils.SpectrogramDataset(data_record, data_input, transforms.ToTensor())

# split data for 80% training and 20% testing
train_size = int(dataset.__len__() * 0.8)
train_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, dataset.__len__() - train_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), 
    lr=learning_rate, 
    momentum=momentum
)

total_steps = len(train_loader)
step_size = ((len(train_set) + 1) // batch_size) // 4
rloss_record = []

# Start training
for epoch in range(num_epochs):
    running_loss = 0.0

    print(f"Epoch {epoch + 1:<3d}")
    for i, data in enumerate(train_loader, 0):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()
        optimizer.step()

        # calculate and print error rates
        running_loss += loss.item()
        utils.displayProgress(i + 1, total_steps, loss.item(), running_loss)

        if (i + 1) % step_size == 0:
            running_loss = 0.0
    
    rloss_record.append(running_loss)
    print()

print("Finished Training \n\n")
torch.save(net.state_dict(), net_output)



# Test CNN accuracy
net_test.load_state_dict(torch.load(net_output))

with torch.no_grad():
    train_preds = utils.get_all_preds(net_test, test_loader, classes)
    test_targets = [dataset.getItemLabel(i) for i in test_set.indices]

    # stack predictions and true labels in an array
    stacked = torch.stack(
        (
            torch.tensor(test_targets),
            torch.max(train_preds, 1)[1]
        ),
        dim=1
    )

    # construct confusion matrix
    cm = torch.zeros(5, 5, dtype=torch.int32)
    for i in stacked:
        x, y = i.tolist()
        cm[x, y] += 1
    
    print(cm)

    if display_conf_matrix:
        df_cm = pd.DataFrame(cm.tolist(), index=[i for i in classes],
                             columns=[i for i in classes])
        
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, fmt="3d", cmap="rocket_r")

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.show()

with open(net_record, "w") as fp:
    for i, loss in enumerate(rloss_record):
        fp.write(f"{i + 1}, {loss}\n")
