"""
Real Estate Valuation - Regression Mini Project (UCI ID 477)
-------------------------------------------------------------
Predict house price per unit area (Sindian District, New Taipei City, Taiwan).

Steps
  1. Load and inspect
  2. EDA (Seaborn)
  3. Preprocessing
  4. Stratified 80/20 split (stratified on price quantile bins)
  5. Train 3+ models (cross-validated)
  6. Evaluate on the test set
  7. Justify the algorithms + interpret the best model
  8. Stacking ensemble

Setup:  pip install pandas numpy scikit-learn seaborn matplotlib ucimlrepo openpyxl
Run:    python real_estate_valuation_master.py
Output: figures saved to ./figures/
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150)
    plt.show()
    plt.close()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


###############################################################################
#1 loading and inspecting

print("=" * 70, "\nSTEP 1: LOAD AND INSPECT\n", "=" * 70, sep="")

COLUMNS = [
    "transaction_date",   # e.g. 2013.250 = March 2013
    "house_age",          # years
    "mrt_distance",       # metres to nearest MRT station
    "convenience_stores", # count within walking distance
    "latitude",
    "longitude",
    "price_per_unit_area",  # target (10,000 New Taiwan Dollar / ping)
]

try:
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=477)
    df = pd.concat([data.data.features, data.data.targets], axis=1)
    df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() == "no"],
                 errors="ignore")
except Exception as exc:  # fallback: local file downloaded from the UCI page
    print(f"ucimlrepo fetch failed ({exc}); trying local file...")
    local = "Real estate valuation data set.xlsx"
    df = pd.read_excel(local)
    df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() == "no"],
                 errors="ignore")

df.columns = COLUMNS  # rename by position (UCI names are long: X1..X6, Y)
TARGET = "price_per_unit_area"

print("Shape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe().T)

###############################################################################
#2 EDA

print("\n" + "=" * 70, "\nSTEP 2: EDA\n", "=" * 70, sep="")

# 2a. Target distribution
plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], kde=True, bins=30, color="steelblue")
plt.title("Distribution of House Price per Unit Area")
save_fig("01_target_distribution")

# 2b. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
save_fig("02_correlation_heatmap")

# 2c. Price vs key features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["mrt_distance", "house_age", "convenience_stores"]):
    sns.scatterplot(data=df, x=col, y=TARGET, alpha=0.6, ax=ax)
    ax.set_title(f"Price vs {col}")
save_fig("03_price_vs_features")

# 2d. MRT distance is heavily skewed -> show raw vs log
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["mrt_distance"], bins=30, ax=axes[0], color="darkorange")
axes[0].set_title("MRT Distance (raw)")
sns.histplot(np.log1p(df["mrt_distance"]), bins=30, ax=axes[1], color="seagreen")
axes[1].set_title("MRT Distance (log1p)")
save_fig("04_mrt_distance_skew")

# 2e. Location map coloured by price
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="longitude", y="latitude", hue=TARGET,
                palette="viridis", size=TARGET, sizes=(15, 120), alpha=0.8)
plt.title("Price by Location")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Price/unit")
save_fig("05_location_price_map")

# 2f. Price by store-count group (boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="convenience_stores", y=TARGET, palette="Blues")
plt.title("Price by Number of Nearby Convenience Stores")
save_fig("06_price_by_stores")


###############################################################################
#3 processing

print("\n" + "=" * 70, "\nSTEP 3: PREPROCESSING\n", "=" * 70, sep="")

# Dataset-specific decisions:
#  - No missing values, so no imputation is needed.
#  - mrt_distance is right-skewed -> log1p transform.
#  - Features are on very different scales (metres vs counts vs degrees)
#    -> StandardScaler (matters for Ridge; harmless for tree models).
#  - Scaling/transforms live inside a Pipeline so they are fit on training
#    data only (no leakage).
#  - Extreme price outliers are KEPT: they are real sales, and tree models
#    handle them; we report MAE alongside RMSE for robustness.
X = df.drop(columns=[TARGET])
y = df[TARGET]

log_cols = ["mrt_distance"]
other_cols = [c for c in X.columns if c not in log_cols]

preprocessor = ColumnTransformer([
    ("log_scale", Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scale", StandardScaler()),
    ]), log_cols),
    ("scale", StandardScaler(), other_cols),
])
print("Features:", list(X.columns))
print("Log-transformed:", log_cols, "| Scaled only:", other_cols)



###############################################################################
#4 stratified 80/20 split

print("\n" + "=" * 70, "\nSTEP 4: STRATIFIED 80/20 SPLIT\n", "=" * 70, sep="")

# Regression target is continuous, so stratify on price quintile bins.
price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=price_bins
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print("Mean price  - train: %.2f | test: %.2f" % (y_train.mean(), y_test.mean()))



###############################################################################
#5 train multiple models, 5 fold CV on the training set


print("\n" + "=" * 70, "\nSTEP 5: TRAIN MODELS\n", "=" * 70, sep="")

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
        random_state=RANDOM_STATE),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
pipelines, cv_rows = {}, []

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    scores = -cross_val_score(pipe, X_train, y_train, cv=cv,
                              scoring="neg_root_mean_squared_error")
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe
    cv_rows.append({"Model": name, "CV RMSE (mean)": scores.mean(),
                    "CV RMSE (std)": scores.std()})
    print(f"{name:<20} CV RMSE = {scores.mean():.3f} (+/- {scores.std():.3f})")



###############################################################################
#6 evaluating on held out test set


print("\n" + "=" * 70, "\nSTEP 6: EVALUATION\n", "=" * 70, sep="")


def evaluate(name, pipe):
    pred = pipe.predict(X_test)
    return {"Model": name,
            "RMSE": rmse(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred)}, pred


results, predictions = [], {}
for name, pipe in pipelines.items():
    row, pred = evaluate(name, pipe)
    results.append(row)
    predictions[name] = pred


###############################################################################
#8 Stacking ensemble (built here so it is included in the comparison)


stack = Pipeline([
    ("prep", preprocessor),
    ("model", StackingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                         random_state=RANDOM_STATE, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                             max_depth=3, subsample=0.8,
                                             random_state=RANDOM_STATE)),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        cv=5, n_jobs=-1)),
])
stack.fit(X_train, y_train)
row, pred = evaluate("Stacking Ensemble", stack)
results.append(row)
predictions["Stacking Ensemble"] = pred
pipelines["Stacking Ensemble"] = stack

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("\nTest-set performance (sorted by RMSE):\n", results_df.round(3).to_string(index=False))

best_name = results_df.loc[0, "Model"]
print(f"\nBest model on test set: {best_name}")



# Model comparison plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
    sns.barplot(data=results_df, x="Model", y=metric, palette="mako", ax=ax)
    ax.set_title(f"Test {metric}")
    ax.tick_params(axis="x", rotation=25)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
save_fig("07_model_comparison")



# Actual vs predicted + residuals for the best model
best_pred = predictions[best_name]
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=y_test, y=best_pred, alpha=0.7, ax=axes[0])
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")
axes[0].set(xlabel="Actual price", ylabel="Predicted price",
            title=f"Actual vs Predicted ({best_name})")
axes[0].legend()
sns.histplot(residuals, kde=True, bins=20, ax=axes[1], color="slateblue")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set(xlabel="Residual (actual - predicted)", title="Residual Distribution")
save_fig("08_best_model_diagnostics")


###############################################################################
#7 Justify algos & interpret


print("\n" + "=" * 70, "\nSTEP 7: ALGORITHM JUSTIFICATION & INTERPRETATION\n", "=" * 70, sep="")

print("""
Why these algorithms?
- Ridge Regression: a transparent linear baseline. Prices tend to move roughly
  linearly with log-distance to transit and with age, and L2 regularisation
  handles the correlation between location features (latitude, longitude,
  MRT distance). If a complex model cannot beat Ridge, the complexity is not
  justified.
- Random Forest: captures non-linear effects and interactions (e.g. a house
  near the MRT AND with many stores is worth disproportionately more) with
  little tuning, and is robust to outliers and unscaled features. Averaging
  many deep trees keeps variance low on a small (~330-row) training set.
- Gradient Boosting: builds trees sequentially to fix earlier errors, usually
  the strongest single model on small tabular data. Shallow trees + subsampling
  + a low learning rate limit overfitting.
- Stacking Ensemble: combines the three. The linear model and the two tree
  models make different kinds of errors, so a RidgeCV meta-learner trained on
  out-of-fold predictions can weight them and often reduces error further.
""")

# Permutation importance on the best model (works for any model type)
perm = permutation_importance(pipelines[best_name], X_test, y_test, n_repeats=30,
                              random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
imp_df = (pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean,
                        "Std": perm.importances_std})
          .sort_values("Importance", ascending=False))
print(f"Permutation importance ({best_name}, increase in RMSE when shuffled):\n",
      imp_df.round(3).to_string(index=False))

plt.figure(figsize=(8, 4))
sns.barplot(data=imp_df, x="Importance", y="Feature", palette="rocket")
plt.title(f"Permutation Importance ({best_name})")
save_fig("09_permutation_importance")

top_feats = ", ".join(imp_df["Feature"].head(2))
print(f"\nBusiness takeaway: the top price drivers are {top_feats}. "
      "Location/transit access and property age explain most of the price "
      "variation, so these should anchor any valuation or pricing advice.")




###############################################################################
#8 Stacking ensemble summary

print("\n" + "=" * 70, "\nSTEP 8: STACKING ENSEMBLE SUMMARY\n", "=" * 70, sep="")

meta = stack.named_steps["model"].final_estimator_
base_names = [n for n, _ in stack.named_steps["model"].estimators]
weights = pd.Series(meta.coef_, index=base_names).round(3)
print("Meta-learner (RidgeCV) weights on base models:\n", weights.to_string())

stack_rmse = results_df.set_index("Model").loc["Stacking Ensemble", "RMSE"]
best_single = results_df[results_df["Model"] != "Stacking Ensemble"].iloc[0]
delta = best_single["RMSE"] - stack_rmse
print(f"\nBest single model: {best_single['Model']} (RMSE {best_single['RMSE']:.3f})")
print(f"Stacking RMSE: {stack_rmse:.3f} -> "
      f"{'improvement' if delta > 0 else 'no improvement'} of {abs(delta):.3f}")

print("\nDone. Figures saved in ./%s/" % FIG_DIR)