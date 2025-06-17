import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from rich.console import Group
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib, os

# Load dataset
df = pd.read_csv('data/training/training_data_adaysn_umap.csv')

X = df.drop('label', axis=1).values
y = df['label'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define DNN model
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x


input_dim = X_train.shape[1]
model = NeuralNet(input_dim)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

progress = Progress(
    TextColumn("[bold blue]Training"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•", TimeRemainingColumn(),
)



def get_metrics_table(loss_val):
    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_row("Current loss", f"{loss_val:.4f}")
    return table


live = Live(
    Group(progress, get_metrics_table(0.0)),  # Start with dummy loss
    refresh_per_second=4,
    transient=False
)

with live:
    task = progress.add_task("", total=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        out = model(X_train)
        loss = criterion(out, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.update(task, advance=1)


        live.update(Group(progress, get_metrics_table(loss.item())))

# Evaluate the Model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_labels = (y_pred >= 0.5).float()

    print("\nClassification Report")
    print(classification_report(y_test, y_pred_labels, target_names=['Benign', 'Malicious']))

    # Confusion Matrix
    confusion_matrix = confusion_matrix(y_test, y_pred_labels)
    print(confusion_matrix)

    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {auc:.4f}")

# Print the results
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"Accuracy: {accuracy:.4f}")



# Save the model
torch.save(model.state_dict(), 'models/dnn_model.pth')
print("Model saved successfully")