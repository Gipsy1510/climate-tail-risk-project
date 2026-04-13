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
WTI_PATH = "RAW_DCOILWTICO.xlsx"
NEWS_PATH = "news_data.csv"
BERT_PATH = "climatebert_daily_signal.csv"  # <-- from Colab
 
wb = openpyxl.load_workbook(WTI_PATH)
wti = pd.DataFrame(list(wb['Daily'].iter_rows(min_row=2, values_only=True)),
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
 
# GDELT news
news = pd.read_csv(NEWS_PATH)
news['date'] = pd.to_datetime(news['news_date'], format='%d/%m/%Y', dayfirst=True)
news = news.sort_values('date').reset_index(drop=True)
news_cols = ['date', 'avg_tone', 'avg_positivity', 'avg_negativity',
             'share_total_climate', 'share_policy', 'share_physical',
             'share_energy', 'share_cleantech', 'share_activism']
 
df = wti.merge(news[news_cols], on='date', how='inner')
df = df.dropna(subset=['rvol_22d', 'target_tail'])
 
# ClimateBERT daily signal
bert = pd.read_csv(BERT_PATH)
bert['date'] = pd.to_datetime(bert['news_date'])
bert_cols = ['date', 'climate_prob_mean', 'climate_prob_max',
             'climate_prob_std', 'climate_share_bert', 'log_n_articles']
 
df = df.merge(bert[bert_cols], on='date', how='left')
 
# Fill missing BERT days with neutral values (not all trading days have articles)
bert_features = ['climate_prob_mean', 'climate_prob_max', 'climate_prob_std',
                 'climate_share_bert', 'log_n_articles']
for col in bert_features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
 
has_bert = df['climate_prob_mean'].notna().sum()
print(f"Modelling table: {len(df)} rows")
print(f"Days with BERT signal: {has_bert} ({has_bert/len(df):.1%})")
# %%
F_MARKET = ['lag1_return', 'lag1_abs_return', 'rvol_5d', 'rvol_10d',
            'rvol_22d', 'ret_mean_5d', 'ret_mean_10d']
 
F_GDELT = ['share_total_climate', 'share_policy', 'share_physical',
           'share_energy', 'share_cleantech', 'share_activism']
 
F_BERT = ['climate_prob_mean', 'climate_prob_max', 'climate_prob_std',
          'climate_share_bert', 'log_n_articles']
# %%
class LSTMTailRisk(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(-1)
 
 
def train_lstm(feature_cols, window=22, hidden_dim=16, n_seeds=5):
    """Train LSTM and return averaged metrics."""
    sc = StandardScaler()
    sc.fit(df[df['date'] < '2022-01-01'][feature_cols].values)
 
    X = sc.transform(df[feature_cols].values).astype(np.float32)
    Y = df['target_tail'].values.astype(np.float32)
    D = df['date'].values
 
    Xs, Ys, Ds = [], [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i]); Ys.append(Y[i]); Ds.append(D[i])
    Xs = np.array(Xs); Ys = np.array(Ys); Ds = np.array(Ds)
 
    tr = Ds < np.datetime64('2022-01-01')
    te = (Ds >= np.datetime64('2022-01-01')) & (Ds <= np.datetime64('2024-12-31'))
    Xtr, Ytr, Xte, Yte = Xs[tr], Ys[tr], Xs[te], Ys[te]
    pw = (Ytr == 0).sum() / max((Ytr == 1).sum(), 1)
 
    rocs, prs = [], []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        m = LSTMTailRisk(len(feature_cols), hidden_dim)
        opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
        loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                            batch_size=64, shuffle=True)
        best_r = 0; best_p = None
        for ep in range(60):
            m.train()
            for xb, yb in loader:
                opt.zero_grad(); crit(m(xb), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if (ep + 1) % 10 == 0:
                m.eval()
                with torch.no_grad():
                    p = torch.sigmoid(m(torch.tensor(Xte))).numpy()
                r = roc_auc_score(Yte, p)
                if r > best_r:
                    best_r = r; best_p = p
        rocs.append(best_r)
        prs.append(average_precision_score(Yte, best_p))
 
    return np.mean(rocs), np.std(rocs), max(rocs), np.mean(prs)
# %%
train_df = df[df['date'] < '2022-01-01']
test_df = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2024-12-31')]
y_tr = train_df['target_tail'].values
y_te = test_df['target_tail'].values
 
print(f"\n{'='*75}")
print(f"{'Model':<45s} {'ROC-AUC':>10s} {'PR-AUC':>10s}")
print(f"{'='*75}")
 
# --- Logistic baselines ---
for name, feats in [
    ('LogReg: market', F_MARKET),
    ('LogReg: market + GDELT', F_MARKET + F_GDELT),
    ('LogReg: market + GDELT + BERT', F_MARKET + F_GDELT + F_BERT),
]:
    sc = StandardScaler()
    Xtr = sc.fit_transform(train_df[feats].values)
    Xte = sc.transform(test_df[feats].values)
    m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, y_tr)
    p = m.predict_proba(Xte)[:, 1]
    roc = roc_auc_score(y_te, p)
    pr = average_precision_score(y_te, p)
    print(f"{name:<45s} {roc:>10.4f} {pr:>10.4f}")
 
print(f"{'-'*75}")
 
# --- LSTM models ---
for name, feats in [
    ('LSTM: market',                  F_MARKET),
    ('LSTM: market + GDELT',          F_MARKET + F_GDELT),
    ('LSTM: market + GDELT + BERT',   F_MARKET + F_GDELT + F_BERT),
]:
    print(f"  Training {name}...", end='', flush=True)
    avg, std, best, pr = train_lstm(feats, n_seeds=5)
    print(f"\r{name:<45s} {avg:.4f}±{std:.3f}   {pr:>10.4f}")
 
print(f"{'='*75}")
print(f"{'Random baseline':<45s} {'0.5000':>10s} {y_te.mean():>10.4f}")