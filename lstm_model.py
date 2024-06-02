import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from gan import *
from process import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('./data/sh600000.csv')
data = process_data(data)
data.drop(['date','ticker','qfq_factor'], axis=1, inplace=True)

X_train = np.array(data.iloc[:, :-1].values)

gandataset = GansDataset(X_train)
gandataloader = DataLoader(gandataset, batch_size=32, shuffle=True)

size = X_train.shape[1]
generator = Generator(size, size)
discriminator = Discriminator(size)

X_train = train_gan(gandataloader, generator, discriminator, size, device, epochs=20)

X_train = np.array(X_train)
y_train = np.array(data.iloc[:, -1].values)

print(X_train.shape)
print(X_train)

class LstmDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 确保标签是序列
        return {'features': torch.FloatTensor(self.features[idx]), 'labels': torch.FloatTensor([self.labels[idx]])}

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, sequence_length, hidden_dim)
        energy = self.attention_linear(lstm_outputs)  # (batch_size, sequence_length, 1)
        weights = F.softmax(energy.squeeze(2), dim=1)  # (batch_size, sequence_length)
        weighted = torch.bmm(weights.unsqueeze(1), lstm_outputs)  # (batch_size, 1, hidden_dim)
        weighted = weighted.squeeze(1)  # (batch_size, hidden_dim)
        return weighted, weights

class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        attn_output, attn_weights = self.attention(lstm_out)
        output = self.fc(attn_output)
        return output

x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_data = LstmDataset(x_train,y_train)
test_data = LstmDataset(x_test,y_test)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

model = LstmModel(input_dim=14, hidden_dim=100, output_dim=1, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train过程
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_dataloader:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(features.unsqueeze(1))
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}')

# Validation过程
model.eval()
val_loss = 0.0
predictions = []
ground_truths = []
with torch.no_grad():
    for batch in test_dataloader:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(features.unsqueeze(1))
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        predictions.extend(outputs.view(-1).cpu().numpy())
        ground_truths.extend(labels.view(-1).cpu().numpy())

print(f'Validation Loss: {val_loss / len(test_dataloader)}')
mse = mean_squared_error(ground_truths, predictions)
r2 = r2_score(ground_truths, predictions)
print(f'Test MSE: {mse}, Test R²: {r2}')

print(predictions)
pd.DataFrame(predictions).to_csv('lstm_test.csv',index=False)