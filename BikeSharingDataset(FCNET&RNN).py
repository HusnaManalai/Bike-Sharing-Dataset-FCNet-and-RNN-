import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Custom data transformations with extensive feature engineering
def custom_transformations(data, epsilon=1e-8):
    # Add polynomial features
    poly_features = np.hstack([data, data**2, data**3])

    # Add interaction terms (pairwise products of features)
    n_features = data.shape[1]
    interaction_features = np.hstack([data[:, i].reshape(-1, 1) * data[:, j].reshape(-1, 1) for i in range(n_features) for j in range(i+1, n_features)])

    # Add logarithmic features
    log_features = np.log1p(data)

    # Combine all features
    combined_features = np.hstack([poly_features, interaction_features, log_features])

    # Add statistical features (mean, variance, skewness, kurtosis)
    mean_features = np.mean(data, axis=1).reshape(-1, 1)
    var_features = np.var(data, axis=1).reshape(-1, 1)
    skew_features = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, data).reshape(-1, 1)
    kurt_features = np.apply_along_axis(lambda x: pd.Series(x).kurt(), 1, data).reshape(-1, 1)
    statistical_features = np.hstack([mean_features, var_features, skew_features, kurt_features])

    # Standardize features
    all_features = np.hstack([combined_features, statistical_features])
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + epsilon  # Add epsilon to avoid division by zero
    standardized_features = (all_features - mean) / std

    return standardized_features

# Define the custom dataset class
class BikeSharingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.features, self.labels = self.preprocess(self.data)

    def preprocess(self, data):
        # Extract features and labels
        y = data['cnt'].values
        X = data.drop(['cnt', 'instant', 'dteday', 'casual', 'registered'], axis=1).values
        
        # Apply custom transformations
        if self.transform:
            X = self.transform(X)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.features[idx], self.labels[idx])
        return sample

# Load dataset
dataset = BikeSharingDataset('hour.csv', transform=custom_transformations)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#FCNet
class BikeSharingFCNet(nn.Module):
    def __init__(self, input_dim):
        super(BikeSharingFCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # 1 output for regression

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

#Recurrent Neural Network
class BikeSharingRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BikeSharingRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 1 output for regression

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
input_dim = dataset.features.shape[1]
model_fc = BikeSharingFCNet(input_dim).to(device)
model_rnn = BikeSharingRNN(input_dim, hidden_dim=64, num_layers=2).to(device)

# Define optimizers
optimizer_fc = optim.Adam(model_fc.parameters(), lr=0.001)
optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)

# Loss function for regression
loss_fn = nn.MSELoss()

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if isinstance(model, BikeSharingRNN):
            data = data.unsqueeze(1)  # Add sequence length dimension for RNN
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if isinstance(model, BikeSharingRNN):
                data = data.unsqueeze(1)  # Add sequence length dimension for RNN
            output = model(data)
            test_loss += loss_fn(output, target.unsqueeze(1)).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

# Train and evaluate the models
for epoch in range(1, 11):
    print(f'\nTraining Fully Connected Network (Epoch {epoch})')
    train(model_fc, device, train_loader, optimizer_fc, epoch)
    test(model_fc, device, test_loader)
    
    print(f'\nTraining Recurrent Neural Network (Epoch {epoch})')
    train(model_rnn, device, train_loader, optimizer_rnn, epoch)
    test(model_rnn, device, test_loader)