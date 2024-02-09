import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class CoordinateDataset(Dataset):
    def __init__(self, coordinates, labels):
        self.coordinates = coordinates
        self.labels = labels

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinate = self.coordinates[idx]
        label = self.labels[idx]
        sample = {
            "coord": torch.tensor(coordinate, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long)
        }
        return sample

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.n1 = nn.Linear(2, 10, bias=False)
        # Changed output dimension to 2 for binary classification
        self.classifier = nn.Linear(12, 1, bias=True)

    def forward(self, coord):
        n1_out = F.relu(self.n1(coord))
        combined = torch.cat((coord, n1_out), dim=1)
        # Removed activation function here to get raw scores (logits)
        out = self.classifier(combined)
        return out

def main(data):
    num_epochs = 500

    coord = data[:, 0:2]
    label = data[:, 2]

    dataset = CoordinateDataset(coord, label)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    model = SimpleMLP()
    # Define an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Use CrossEntropyLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(num_epochs):

        total = 0
        correct = 0

        for batch_idx, sample in enumerate(tqdm(dataloader)):
            inputs = sample["coord"]
            targets = sample["label"].unsqueeze(1)
            targets = targets.to(dtype=torch.float32)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = (torch.sigmoid(outputs) > 0.5).float()

            total += targets.shape[0]
            correct += (preds == targets).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {correct/total:.4f}')

    # # Print all model weights and biases
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")
    #     print(param.data)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'simple_mlp_model.pth')

def create_histogram(array):
    # Plot the histogram
    plt.hist(array, bins='auto', alpha=0.7, rwidth=0.85)  # 'auto' lets matplotlib decide the number of bins
    
    # Add a title and labels (optional, but recommended for clarity)
    plt.title('Histogram')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    
    # Show the plot
    plt.savefig("histogram.png")

if __name__ == "__main__":
    data = np.load("hw1_p2.npy")
    main(data)

    array = np.array([1.0]*10)
    create_histogram(array)

    # array = np.array([1.0,1.0, 0.995, 0.9975, 1.0, 1.0, 0.995, 1.0, 0.9975, 0.9925])
    # create_histogram(array)