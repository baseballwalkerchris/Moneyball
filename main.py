import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('./kaggle/nba_2022-23_all_stats_with_salary.csv', usecols = ["Player Name", "Salary", "PTS", "AST", "DRB", "ORB", "STL", "TOV", "FT", "BLK"])

X = df[["PTS", "AST", "DRB", "ORB", "STL", "TOV", "FT", "BLK"]]
print(X.shape)
y = df["Salary"]
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class SalaryPredict(nn.Module):
    def __init__(self, input_size):
        super(SalaryPredict, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = SalaryPredict(input_size=X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train_tensor)

    # Compute loss
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            y_pred_test = model(X_test_tensor)
            test_loss = criterion(y_pred_test, y_test_tensor)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Evaluate on the test set


# Make predictions for new data
new_data = torch.tensor([[25, 5, 10, 1, 1, 1, 1, 1]], dtype=torch.float32)
new_data_scaled = scaler.transform(new_data.numpy())
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)
prediction = model(new_data_tensor)
print(f'Predicted Salary: {prediction.item()}')
