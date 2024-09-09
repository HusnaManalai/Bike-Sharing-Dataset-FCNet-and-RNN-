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
        
        # Ensure input dimension is divisible by the number of heads (4)
        input_dim = X.shape[1]
        if input_dim % 4 != 0:
            pad_length = 4 - (input_dim % 4)
            X = np.pad(X, ((0, 0), (0, pad_length)), 'constant', constant_values=0)

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


#GRU
class BikeSharingGRUConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(BikeSharingGRUConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.gru = nn.GRU(64, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

#Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a sequence length dimension
        x = x.permute(1, 0, 2)  # Convert to (seq_len, batch, input_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Pooling
        x = self.dropout(x)
        x = self.bn(x)
        out = self.fc(x)
        return out


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
input_dim = dataset.features.shape[1]
# Make sure input_dim is divisible by the number of heads
if input_dim % 4 != 0:
    input_dim += 4 - (input_dim % 4)

model_gru = BikeSharingGRUConvNet(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.5).to(device)
model_transformer = TransformerModel(input_dim=input_dim, num_heads=4, hidden_dim=256, num_layers=3, dropout=0.2).to(device)

# Define optimizers
optimizer_gru = optim.Adam(model_gru.parameters(), lr=0.001)
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=0.001)

# Loss function for regression
loss_fn = nn.MSELoss()

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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
            output = model(data)
            test_loss += loss_fn(output, target.unsqueeze(1)).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

# Train and evaluate the models
for epoch in range(1, 11):
    print(f'\nTraining GRU with Convolutional Layers (Epoch {epoch})')
    train(model_gru, device, train_loader, optimizer_gru, epoch)
    test(model_gru, device, test_loader)
    
    print(f'\nTraining Transformer Encoder with Batch Normalization and ReLU (Epoch {epoch})')
    train(model_transformer, device, train_loader, optimizer_transformer, epoch)
    test(model_transformer, device, test_loader)
