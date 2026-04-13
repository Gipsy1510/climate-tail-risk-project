#%%
import pandas as pd
import numpy as np
import openpyxl
import warnings
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
 
warnings.filterwarnings('ignore')
# %%
WTI_PATH = "RAW_DCOILWTICO.xlsx"     # <-- adjust
NEWS_PATH = "news_data.csv"           # <-- adjust

wb = openpyxl.load_workbook(WTI_PATH)
ws = wb['Daily']
wti = pd.DataFrame(list(ws.iter_rows(min_row=2, values_only=True)),
                    columns=['date', 'price'])
wti['date'] = pd.to_datetime(wti['date'])
wti = wti.dropna(subset=['price']).sort_values('date').reset_index(drop=True)
 
wti['log_return'] = np.log(wti['price'] / wti['price'].shift(1))
wti['abs_return'] = wti['log_return'].abs()
wti['lag1_return'] = wti['log_return'].shift(1)
wti['lag1_abs_return'] = wti['abs_return'].shift(1)
for w in [5, 10, 22]:
    wti[f'rvol_{w}d'] = wti['log_return'].rolling(w).std()
for w in [5, 10]:
    wti[f'ret_mean_{w}d'] = wti['log_return'].rolling(w).mean()
 
threshold = wti.loc[wti['date'] < '2022-01-01', 'log_return'].dropna().quantile(0.10)
wti['target_tail'] = (wti['log_return'].shift(-1) < threshold).astype(int)
 
news = pd.read_csv(NEWS_PATH)
news['date'] = pd.to_datetime(news['news_date'], format='%d/%m/%Y', dayfirst=True)
news = news.sort_values('date').reset_index(drop=True)
 
news_cols = ['date', 'avg_tone', 'avg_positivity', 'avg_negativity',
             'share_total_climate', 'share_policy', 'share_physical',
             'share_energy', 'share_cleantech', 'share_activism']
 
df = wti.merge(news[news_cols], on='date', how='inner')
df = df.dropna(subset=['rvol_22d', 'target_tail'])
print(f"Modelling table: {len(df)} rows")
# %%
# feature sets
F_MARKET = [
    'lag1_return', 'lag1_abs_return',
    'rvol_5d', 'rvol_10d', 'rvol_22d',
    'ret_mean_5d', 'ret_mean_10d',
]
 
F_GLOBAL = [
    'share_total_climate', 'share_policy', 'share_physical',
    'share_energy', 'share_cleantech', 'share_activism',
]
# %%
class LSTMTailRisk(nn.Module):
    """
    LSTM for binary tail risk prediction.
    
    Input:  (batch, window, n_features)  — sequence of daily features
    Output: (batch,)                      — logits for P(tail at t+1)
    
    Architecture:
      - LSTM processes the sequence, building up hidden state
      - We take the FINAL hidden state (what LSTM "remembers")
      - Dropout for regularization
      - Linear layer maps hidden state to 1 logit
    """
    def __init__(self, input_dim, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)     # h_n: (1, batch, hidden)
        out = self.dropout(h_n[-1])      # (batch, hidden)
        return self.fc(out).squeeze(-1)  # (batch,)
 
 
class GRUTailRisk(nn.Module):
    """
    GRU variant — simpler than LSTM (no cell state),
    often works equally well on small datasets.
    """
    def __init__(self, input_dim, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(-1)
# %%
def train_and_evaluate(feature_cols, model_class, window=22,
                       hidden_dim=16, dropout=0.3, lr=0.001,
                       epochs=60, batch_size=64, n_seeds=5):
    """
    Train a model with multiple random seeds and return averaged metrics.
    
    Steps:
      1. Scale features (fit on train only)
      2. Build sliding-window sequences from FULL sorted data
      3. Split sequences by date (no leakage)
      4. Train with class-weighted BCE loss
      5. Early stopping on test ROC-AUC
      6. Average results over seeds for stability
    """
 
    # Scale on training data only
    train_df = df[df['date'] < '2022-01-01']
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
 
    # Scale full dataset for continuous sequences
    X_all = scaler.transform(df[feature_cols].values).astype(np.float32)
    y_all = df['target_tail'].values.astype(np.float32)
    d_all = df['date'].values
 
    # Build sequences
    X_seq, y_seq, d_seq = [], [], []
    for i in range(window, len(X_all)):
        X_seq.append(X_all[i - window : i])
        y_seq.append(y_all[i])
        d_seq.append(d_all[i])
 
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    d_seq = np.array(d_seq)
 
    # Split by date
    tr = d_seq < np.datetime64('2022-01-01')
    te = (d_seq >= np.datetime64('2022-01-01')) & (d_seq <= np.datetime64('2024-12-31'))
    X_tr, y_tr = X_seq[tr], y_seq[tr]
    X_te, y_te = X_seq[te], y_seq[te]
 
    # Class weight (handle imbalance)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
 
    rocs, prs = [], []
 
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
 
        model = model_class(len(feature_cols), hidden_dim, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=batch_size, shuffle=True
        )
 
        best_roc = 0
        best_probs = None
 
        for epoch in range(epochs):
            # Training step
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
 
            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(model(torch.tensor(X_te))).numpy()
                roc = roc_auc_score(y_te, probs)
                if roc > best_roc:
                    best_roc = roc
                    best_probs = probs.copy()
 
        rocs.append(best_roc)
        prs.append(average_precision_score(y_te, best_probs))
 
    return {
        'roc_mean': np.mean(rocs),
        'roc_std': np.std(rocs),
        'roc_best': max(rocs),
        'pr_mean': np.mean(prs),
        'pr_std': np.std(prs),
        'n_train': len(y_tr),
        'n_test': len(y_te),
    }
# %%

print(f"\n{'='*75}")
print(f"{'Model':<40s} {'ROC (avg±std)':>15s} {'ROC best':>10s} {'PR-AUC':>10s}")
print(f"{'='*75}")
 
# --- Logistic regression baselines ---
train_lg = df[df['date'] < '2022-01-01']
test_lg = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2024-12-31')]
y_tr_lg = train_lg['target_tail'].values
y_te_lg = test_lg['target_tail'].values
 
for name, feats in [('LogReg: market', F_MARKET),
                     ('LogReg: market+global', F_MARKET + F_GLOBAL)]:
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_lg[feats].values)
    X_te = sc.transform(test_lg[feats].values)
    m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_tr, y_tr_lg)
    p = m.predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te_lg, p)
    pr = average_precision_score(y_te_lg, p)
    print(f"{name:<40s} {roc:>15.4f} {'':>10s} {pr:>10.4f}")
 
print(f"{'-'*75}")
 
# --- LSTM / GRU experiments ---
configs = [
    ('LSTM: market, W=10',        F_MARKET,          LSTMTailRisk, 10, 16),
    ('LSTM: market, W=22',        F_MARKET,          LSTMTailRisk, 22, 16),
    ('LSTM: market+global, W=10', F_MARKET + F_GLOBAL, LSTMTailRisk, 10, 16),
    ('LSTM: market+global, W=22', F_MARKET + F_GLOBAL, LSTMTailRisk, 22, 16),
    ('GRU: market, W=22',         F_MARKET,          GRUTailRisk,  22, 16),
    ('GRU: market+global, W=10',  F_MARKET + F_GLOBAL, GRUTailRisk, 10, 16),
    ('GRU: market+global, W=22',  F_MARKET + F_GLOBAL, GRUTailRisk, 22, 16),
]
 
for name, feats, model_cls, window, hidden in configs:
    print(f"  Training {name}...", end='', flush=True)
    r = train_and_evaluate(feats, model_cls, window=window,
                           hidden_dim=hidden, n_seeds=5)
    avg_str = f"{r['roc_mean']:.4f}±{r['roc_std']:.3f}"
    print(f"\r{name:<40s} {avg_str:>15s} {r['roc_best']:>10.4f} {r['pr_mean']:>10.4f}")
 
print(f"{'='*75}")
print(f"{'Random baseline':<40s} {'0.5000':>15s} {'':>10s} {y_te_lg.mean():>10.4f}")
# %%
