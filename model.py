# model.py
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # out: (batch, seq_len, hidden)
        out = out[:, -1, :]        # last time step
        out = self.fc(out)         # (batch, 1)
        return out

def train_model(model, train_loader, num_epochs=10, lr=1e-3, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, loss={epoch_loss:.6f}")

    return model

def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy()
    return preds
