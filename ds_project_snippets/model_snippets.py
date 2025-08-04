# Data preprocessing snippet
for df in [train, test]:
    df['Agent'] = df['Agent'].fillna(0).astype(int).astype('object')
    df['Company'] = df['Company'].fillna(0).astype(int).astype('object')

# fweature engineering & encoding
combined = pd.concat([X, test], keys=['train', 'test'])
cat_cols = combined.select_dtypes(include='object').columns.tolist()
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
combined_scaled = combined_encoded.apply(lambda x: (x - np.mean(x)) / np.std(x))

# LASSO logistic regr with grid search
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}
grid = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear'), param_grid, cv=10, n_jobs=-1)
grid.fit(X_encoded, y)

print("Best C:", grid.best_params_['C'])
model = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', C=0.03)
model.fit(X_encoded, y)
