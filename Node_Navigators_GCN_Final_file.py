

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import json

# =========================
# 1. Load Data
# =========================

# Load adjacency matrix, features, labels, and splits
adj = sp.load_npz('/Users/basusmac/Desktop/Github Repositories/Data_Mining/data/data/adj.npz')
feat = np.load('/Users/basusmac/Desktop/Github Repositories/Data_Mining/data/data/features.npy')
labels = np.load('/Users/basusmac/Desktop/Github Repositories/Data_Mining/data/data/labels.npy')
splits = json.load(open('/Users/basusmac/Desktop/Github Repositories/Data_Mining/data/data/splits.json'))
idx_train, idx_test = splits['idx_train'], splits['idx_test']

# Convert to PyTorch tensors
x = torch.from_numpy(feat).float()
edge_index, _ = from_scipy_sparse_matrix(adj)
y = torch.full((x.shape[0],), -1, dtype=torch.long)  # initialize all labels as -1
y[idx_train] = torch.from_numpy(labels).long()       # set train labels

# =========================
# 2. Define GCN Model
# =========================

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# =========================
# 3. Training Setup
# =========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
idx_train = torch.tensor(idx_train, dtype=torch.long, device=device)
idx_test = torch.tensor(idx_test, dtype=torch.long, device=device)

num_features = x.shape[1]
num_classes = labels.max() + 1

model = GCN(num_features, hidden_dim=64, num_classes=num_classes, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# =========================
# 4. Training Loop
# =========================

best_acc = 0
best_pred = None

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[idx_train], y[idx_train])
    loss.backward()
    optimizer.step()

    # Validation on train set (since no val set)
    model.eval()
    _, pred = out.max(dim=1)
    correct = int((pred[idx_train] == y[idx_train]).sum())
    acc = correct / len(idx_train)
    if acc > best_acc:
        best_acc = acc
        best_pred = pred.detach().cpu().numpy()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train Acc: {acc*100:.2f}%")

# =========================
# 5. Predict and Save Results
# =========================

model.eval()
with torch.no_grad():
    out = model(x, edge_index)
    pred = out.argmax(dim=1).cpu().numpy()

# Ensure the output directory exists
output_dir = '/Users/basusmac/Desktop/Github Repositories/Data_Mining'
os.makedirs(output_dir, exist_ok=True)

# Get the test node indices
test_node_ids = idx_test.cpu().numpy()

# Stack node IDs and predictions
submission = np.stack([test_node_ids, pred[test_node_ids]], axis=1)

# Save the submission with node IDs
output_path = os.path.join(output_dir, 'your_team_submission_2.txt')
np.savetxt(output_path, submission, fmt='%d %d')  # Two integers: node ID and prediction

# Print confirmation and show the first 10 lines
print(f"Submission saved to: {output_path}")
with open(output_path, 'r') as f:
    for _ in range(10):
        print(f.readline().strip())

