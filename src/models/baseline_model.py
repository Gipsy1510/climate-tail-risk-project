#%%
import pandas as pd
import numpy as np
import openpyxl
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# %%
WTI_PATH = "RAW_DCOILWTICO.xlsx"         # <-- adjust path
NEWS_PATH = "news_data.csv"               # <-- adjust path
 
wb = openpyxl.load_workbook(WTI_PATH)
ws = wb['Daily']
rows = list(ws.iter_rows(min_row=2, values_only=True))
wti = pd.DataFrame(rows, columns=['date', 'price'])
wti['date'] = pd.to_datetime(wti['date'])
wti = wti.dropna(subset=['price']).sort_values('date').reset_index(drop=True)
print(f"WTI: {len(wti)} trading days ({wti['date'].min().date()} → {wti['date'].max().date()})")
# %%

#Market features
wti['log_return']      = np.log(wti['price'] / wti['price'].shift(1))
wti['abs_return']      = wti['log_return'].abs()
wti['lag1_return']     = wti['log_return'].shift(1)
wti['lag1_abs_return'] = wti['abs_return'].shift(1)
 
for w in [5, 10, 22]:
    wti[f'rvol_{w}d'] = wti['log_return'].rolling(w).std()
for w in [5, 10]:
    wti[f'ret_mean_{w}d'] = wti['log_return'].rolling(w).mean()
 
# Target: next-day return below 10th percentile (training period only)
threshold = wti.loc[wti['date'] < '2022-01-01', 'log_return'].dropna().quantile(0.10)
wti['target_tail'] = (wti['log_return'].shift(-1) < threshold).astype(int)
print(f"Tail threshold: {threshold:.4f} ({threshold*100:.2f}%)")
# %%

# News features
news = pd.read_csv(NEWS_PATH, parse_dates=['news_date'])
news['date'] = pd.to_datetime(news['news_date'], format='%d/%m/%Y', dayfirst=True)
news = news.sort_values('date').reset_index(drop=True)
 
# Within-climate shares
for label, raw in [
    ('share_policy_within',   'n_policy_regulation'),
    ('share_physical_within', 'n_physical_risk'),
    ('share_energy_within',   'n_energy_geopolitics'),
    ('share_cleantech_within','n_cleantech'),
    ('share_activism_within', 'n_activism_litigation'),
]:
    news[label] = news[raw] / news['n_articles_climate']
 
# Narrative entropy
buckets = ['n_core_climate', 'n_policy_regulation', 'n_physical_risk',
           'n_energy_geopolitics', 'n_cleantech', 'n_activism_litigation']
props = news[buckets].div(news[buckets].sum(axis=1), axis=0)
news['narrative_entropy'] = -(props * np.log(props + 1e-10)).sum(axis=1)
 
print(f"News: {len(news)} days, {news.isnull().sum().sum()} NaN")
# %%

#merge
news_cols = [
    'date', 'log_n_articles_climate',
    'avg_tone', 'avg_positivity', 'avg_negativity',
    'share_total_climate', 'share_policy', 'share_physical',
    'share_energy', 'share_cleantech', 'share_activism',
    'share_policy_within', 'share_physical_within', 'share_energy_within',
    'share_cleantech_within', 'share_activism_within', 'narrative_entropy',
]
 
df = wti.merge(news[news_cols], on='date', how='inner')
df = df.dropna(subset=['rvol_22d', 'target_tail'])
print(f"Modelling table: {len(df)} rows")
# %%

#split train and test
train = df[df['date'] < '2022-01-01'].copy()
test  = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2024-12-31')].copy()
y_train = train['target_tail'].values
y_test  = test['target_tail'].values
 
print(f"Train: {len(train)} rows ({train['target_tail'].mean():.2%} tail rate)")
print(f"Test:  {len(test)} rows ({test['target_tail'].mean():.2%} tail rate)")
# %%

#Feature sets
F_market = [
    'lag1_return', 'lag1_abs_return',
    'rvol_5d', 'rvol_10d', 'rvol_22d',
    'ret_mean_5d', 'ret_mean_10d',
]
 
F_news_within = [
    'log_n_articles_climate',
    'avg_tone', 'avg_positivity', 'avg_negativity',
    'narrative_entropy',
    'share_policy_within', 'share_physical_within',
    'share_energy_within', 'share_cleantech_within',
    'share_activism_within',
]
 
F_news_global = [
    'share_total_climate', 'share_policy', 'share_physical',
    'share_energy', 'share_cleantech', 'share_activism',
]


# %%
#ALL MODELS -----------------------------------------------------------------
configs = {
    'M1: Market-only':          F_market,
    'M2: Market+News(within)':  F_market + F_news_within,
    'M3: Market+News(global)':  F_market + F_news_global,
    'M4: Market+News(all)':     F_market + F_news_within + F_news_global,
    'M5: News-only(within)':    F_news_within,
    'M6: News-only(global)':    F_news_global,
}
 
results = {}
baseline_roc = None
 
print(f"\n{'Model':<30s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Δ ROC':>8s}")
print("-" * 60)
 
for name, feats in configs.items():
    sc = StandardScaler()
    X_tr = sc.fit_transform(train[feats].values)
    X_te = sc.transform(test[feats].values)
 
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_tr, y_train)
    probs = model.predict_proba(X_te)[:, 1]
 
    roc = roc_auc_score(y_test, probs)
    pr  = average_precision_score(y_test, probs)
 
    if baseline_roc is None:
        baseline_roc = roc
 
    results[name] = {'roc': roc, 'pr': pr, 'model': model, 'feats': feats, 'probs': probs}
    print(f"{name:<30s} {roc:>9.4f} {pr:>9.4f} {roc - baseline_roc:>+8.4f}")
 
#----------------------------------------------------------------


# %%
#LASSO regularization
all_feats = F_market + F_news_within + F_news_global
sc = StandardScaler()
X_tr = sc.fit_transform(train[all_feats].values)
X_te = sc.transform(test[all_feats].values)
 
lasso = LogisticRegressionCV(
    penalty='l1', solver='saga', Cs=20, cv=5,
    class_weight='balanced', max_iter=2000, scoring='roc_auc',
)
lasso.fit(X_tr, y_train)
probs_l = lasso.predict_proba(X_te)[:, 1]
roc_l = roc_auc_score(y_test, probs_l)
pr_l  = average_precision_score(y_test, probs_l)
print(f"{'Lasso (all features)':<30s} {roc_l:>9.4f} {pr_l:>9.4f} {roc_l - baseline_roc:>+8.4f}")
 
print(f"\nLasso selected features:")
for feat, coef in sorted(zip(all_feats, lasso.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    if abs(coef) > 0.001:
        print(f"  {feat:<30s} {coef:>+.4f}")
# %%

#BEST MODEL ANALYSIS ------------------------------------------------------
best_name = max(results, key=lambda k: results[k]['roc'])
best = results[best_name]
print(f"\nBest model: {best_name} (ROC={best['roc']:.4f})")
print(f"  Coefficients (sorted by importance):")
for feat, coef in sorted(zip(best['feats'], best['model'].coef_[0]),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"    {feat:<30s} {coef:>+.4f}  {'↑' if coef > 0 else '↓'} tail risk")


# %%
#Regime stress analysis
for period, mask in [("2022 (stress)", test['date'].dt.year == 2022),
                     ("2023-24 (calm)", test['date'].dt.year >= 2023)]:
    y_sub = y_test[mask.values]
    p_sub = best['probs'][mask.values]
    if y_sub.sum() > 0:
        print(f"  {period}: ROC={roc_auc_score(y_sub, p_sub):.4f}  "
              f"PR={average_precision_score(y_sub, p_sub):.4f}  "
              f"(n={len(y_sub)}, tails={y_sub.sum()})")
# %%

# NOTES:M3 (Market+News global shares) performs best with M1 too.
# The global shares measure how prominent climate news is in the overall media landscape — that's a signal.
# M2 axtually hurts - 10 extra features describing the composition of climate news add noise
# Lasso regression also selects mostly global share features, reinforcing this finding. The model has better performance during the 2022 stress period than in the calmer 2023-24 period, indicating it captures tail risk better when it's more pronounced.
# IN LASSO, when you use all features, it keeps only one. It tells that in a linear framework, the news features are too noisy to survive regularization.
# Overall, logistic regression can only capture today's snapshot. It cant see that volatility has been climbing for 5 consecutive days.
# =>> LSTM