import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cnn_utils as cnu
import cnn_models as cnm
import pandas as pd
import seaborn as sn


def get_all_preds(net, loader, classes):
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

    for classname, count in correct_pred.items():
        accuracy = 100 * float(count) / total_pred[classname]
        print(
            f"Accuracy {classname:4s}: {count} / {total_pred[classname]} = {accuracy:.2f}%")

    return all_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
batch_size = 32
learning_rate = 0.001
momentum = 0.9

classes = ("FLAC", "V0", "320K", "192K", "128K")

dataset = cnu.SpectrogramDataset(
    "./data/dataset_3110/spectral_record.csv", "./data/dataset_3110/spectrograms/", transforms.ToTensor())

train_size = int(dataset.__len__() * 0.8)

train_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, dataset.__len__() - train_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


net = cnm.ConvNetEB().to(device)
net_test = cnm.ConvNetEB()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=learning_rate, momentum=momentum)

step_size = ((len(train_set) + 1) // batch_size) // 4

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % step_size == 0:
            print(
                f"[Epoch {epoch + 1:3d}, Step {i + 1:3d}] loss: {running_loss / step_size:.5f}")
            running_loss = 0.0

print("Finished Training\n\n")
torch.save(net.state_dict(), "./spectral_net.pt")

net_test.load_state_dict(torch.load("./spectral_net.pt"))

with torch.no_grad():
    train_preds = get_all_preds(net, test_loader, classes)
    test_targets = [dataset.getItemLabel(i) for i in test_set.indices]

    stacked = torch.stack(
        (
            torch.tensor(test_targets),
            torch.max(train_preds, 1)[1]
        ),
        dim=1
    )

    cm = torch.zeros(5, 5, dtype=torch.int32)

    for p in stacked:
        j, k = p.tolist()
        cm[j, k] += 1

    print(cm)
    """
    df_cm = pd.DataFrame(cm.tolist(), index=[i for i in classes], columns=[i for i in classes])
    
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="3d", cmap="rocket_r")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.show()
    """
