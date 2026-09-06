# %% [markdown]
# UCI 477 Real Estate Valuation Pipeline

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
TEST_SIZE = 1 / 3
CV_FOLDS = 5
CV_SPLITTER = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
CV_SELECT_METRIC = "cv_rmse"
CV_SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
X3_COL = "X3 distance to the nearest MRT station"
pd.set_option("display.width", 10000)
pd.set_option("display.max_columns", None)

# %%
ds = fetch_ucirepo(id=477)
print("features:", ds.data.features.columns.tolist())
print("targets:", ds.data.targets.columns.tolist())
print("ids:", ds.data.ids.columns.tolist())
X = ds.data.features.copy()
y = ds.data.targets.iloc[:, 0]

# %%
df = X.copy()
df["Y"] = y.values
print("shape:", df.shape)
print("missing:", df.isna().sum().sum())
print("duplicates:", df.duplicated().sum())
print("describe:\n", df.describe())

# %%
feature_cols = [c for c in df.columns if c != "Y"]
X_all, y_all = df[feature_cols], df["Y"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train = X_train.copy()
X_test = X_test.copy()
X_train["X3_log"] = np.log1p(X_train[X3_COL])
X_test["X3_log"] = np.log1p(X_test[X3_COL])
feature_sets = {"base": feature_cols, "base_x3log": feature_cols + ["X3_log"]}
print(f"train={X_train.shape[0]} test={X_test.shape[0]}")

# %%
train_df = X_train.copy()
train_df["Y"] = y_train
print("\ndescribe:\n", train_df.describe())
print("\ncorrelation:\n", train_df.corr())
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.flat, train_df.columns):
    ax.hist(train_df[col], bins=20, edgecolor="black")
    ax.set_title(col, fontsize=8)
for ax in axes.flat[len(train_df.columns):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=100)
plt.close()
corr = train_df.corr()
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=7)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=100)
plt.close()
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for ax, col in zip(axes.flat, feature_cols):
    ax.scatter(train_df[col], train_df["Y"], s=8, alpha=0.6)
    ax.set_title(col, fontsize=8)
plt.tight_layout()
plt.savefig("eda_scatter.png", dpi=100)
plt.close()

# %%
models = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "svr": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
}
cv_rows = []
for feat_name, cols in feature_sets.items():
    Xt = X_train[cols]
    for target_name, use_log in (("Y", False), ("Y_log", True)):
        for model_name, model in models.items():
            est = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1) if use_log else model
            s = cross_validate(est, Xt, y_train, cv=CV_SPLITTER, scoring=CV_SCORING, n_jobs=-1)
            cv_rows.append({"features": feat_name, "target": target_name, "model": model_name, "cv_mae": -s["test_mae"].mean(), "cv_rmse": -s["test_rmse"].mean(), "cv_r2": s["test_r2"].mean()})
cv_results = pd.DataFrame(cv_rows).sort_values(CV_SELECT_METRIC, ascending=True)
print(cv_results.round(3).to_string(index=False))

# %%
best = cv_results.iloc[0]
best_cols = feature_sets[best["features"]]
best_model = models[best["model"]]
best_est = TransformedTargetRegressor(regressor=best_model, func=np.log1p, inverse_func=np.expm1) if best["target"] == "Y_log" else best_model
best_est.fit(X_train[best_cols], y_train)
test_pred = best_est.predict(X_test[best_cols])
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
test_r2 = r2_score(y_test, test_pred)
print(f"best features={best['features']} target={best['target']} model={best['model']} test_mae={test_mae:.3f} test_rmse={test_rmse:.3f} test_r2={test_r2:.3f}")
