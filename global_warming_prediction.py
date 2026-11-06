"""
=====================================================================
  GLOBAL WARMING PREDICTION USING CLIMATE DATA OF ALL COUNTRIES
  Data Science & Analytics Project
=====================================================================
Dataset : Berkeley Earth - "GlobalLandTemperaturesByCountry.csv"
          (Kaggle: Climate Change: Earth Surface Temperature Data)
          ~577,000 rows, 243 countries, monthly data 1743-2013

Goal    : Predict the average monthly land temperature of a country
          from Year, Month and Country information.

Models  : 1) Linear Regression        (simple, explains the warming trend)
          2) Random Forest Regressor  (powerful, captures complex patterns)

Split   : 80% training data, 20% testing data

Metrics : Regression     -> MAE, MSE, RMSE, R² Score, Prediction Accuracy
          Classification -> Accuracy, Precision, Recall, F1 Score,
                            Confusion Matrix (temperature grouped into
                            Cold / Moderate / Hot)
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# -------------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# Keep the CSV file in the SAME folder as this .py file
# -------------------------------------------------------------------
print("STEP 1: Loading dataset...")
df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
print("Total rows in original dataset :", len(df))
print("Total columns                  :", len(df.columns))
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nStatistical summary:")
print(df.describe())

# -------------------------------------------------------------------
# STEP 2: DATA CLEANING & PREPARATION
# -------------------------------------------------------------------
print("\nSTEP 2: Cleaning data...")

# Remove rows where temperature is missing
df = df.dropna(subset=["AverageTemperature"])

# Convert date column into Year and Month
df["dt"] = pd.to_datetime(df["dt"])
df["Year"] = df["dt"].dt.year
df["Month"] = df["dt"].dt.month

# Use data from 1900 onwards (older readings are less reliable)
df = df[df["Year"] >= 1900]

# Convert Country names into numbers (computers need numbers)
df["Country_Code"] = df["Country"].astype("category").cat.codes

print("Rows after cleaning           :", len(df))
print("Number of countries           :", df["Country"].nunique())

# -------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------
print("\nSTEP 3: Exploratory Data Analysis...")

# 3a. Is the world getting warmer?
yearly = df.groupby("Year")["AverageTemperature"].mean()
plt.figure(figsize=(10, 5))
plt.plot(yearly.index, yearly.values, color="orange", label="Yearly average")
plt.plot(yearly.index, yearly.rolling(10).mean(), color="red",
         linewidth=2, label="10-year moving average")
plt.title("Global Average Land Temperature (1900 - 2013)")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("1_global_temperature_trend.png")
plt.show()

# 3b. Correlation heatmap
corr_cols = ["AverageTemperature", "AverageTemperatureUncertainty",
             "Year", "Month", "Country_Code"]
corr = df[corr_cols].corr()
print("\nCorrelation matrix:")
print(corr.round(3))

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("2_correlation_heatmap.png")
plt.show()

# -------------------------------------------------------------------
# STEP 4: SPLIT DATA - 80% TRAINING, 20% TESTING
# -------------------------------------------------------------------
print("\nSTEP 4: Splitting data into 80% train and 20% test...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Feature engineering: "how hot is this country normally?"
# Calculated ONLY from training data so the test data stays unseen.
country_avg = train_df.groupby("Country")["AverageTemperature"].mean()
overall_avg = train_df["AverageTemperature"].mean()
train_df = train_df.copy()
test_df = test_df.copy()
train_df["Country_AvgTemp"] = train_df["Country"].map(country_avg)
test_df["Country_AvgTemp"] = test_df["Country"].map(country_avg).fillna(overall_avg)

features = ["Year", "Month", "Country_Code", "Country_AvgTemp"]
target = "AverageTemperature"

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print("Training rows (80%) :", len(X_train))
print("Testing rows  (20%) :", len(X_test))

# -------------------------------------------------------------------
# STEP 5: ALGORITHM 1 - LINEAR REGRESSION
# -------------------------------------------------------------------
print("\nSTEP 5: Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# -------------------------------------------------------------------
# STEP 6: ALGORITHM 2 - RANDOM FOREST REGRESSOR
# (takes about 1-3 minutes, please wait)
# -------------------------------------------------------------------
print("STEP 6: Training Random Forest (please wait 1-3 minutes)...")
rf_model = RandomForestRegressor(n_estimators=50, max_depth=20,
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# -------------------------------------------------------------------
# STEP 7: EVALUATE BOTH MODELS
# -------------------------------------------------------------------
# To calculate Accuracy, Precision, Recall and F1 Score, the temperature
# is grouped into 3 classes:
#   Cold     : below 10 °C
#   Moderate : 10 °C to 25 °C
#   Hot      : above 25 °C
labels = ["Cold", "Moderate", "Hot"]

def to_class(temps):
    return pd.cut(temps, bins=[-np.inf, 10, 25, np.inf], labels=labels).astype(str)

y_test_class = to_class(np.asarray(y_test))

def evaluate(name, y_true, y_pred):
    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Prediction accuracy: % of predictions within ±2 °C of the real value
    pred_acc = np.mean(np.abs(np.asarray(y_true) - y_pred) <= 2)

    # Classification metrics (Cold / Moderate / Hot)
    pred_class = to_class(y_pred)
    acc = accuracy_score(y_test_class, pred_class)
    prec = precision_score(y_test_class, pred_class, average="weighted", zero_division=0)
    rec = recall_score(y_test_class, pred_class, average="weighted", zero_division=0)
    f1 = f1_score(y_test_class, pred_class, average="weighted", zero_division=0)

    print(f"\n{'=' * 55}\n  {name}\n{'=' * 55}")
    print("  REGRESSION METRICS")
    print(f"  MAE  (Mean Absolute Error)      : {mae:.3f} °C")
    print(f"  MSE  (Mean Squared Error)       : {mse:.3f}")
    print(f"  RMSE (Root Mean Squared Error)  : {rmse:.3f} °C")
    print(f"  R² Score                        : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"  Prediction Accuracy (±2 °C)     : {pred_acc*100:.2f}%")
    print("\n  CLASSIFICATION METRICS (Cold / Moderate / Hot)")
    print(f"  Accuracy                        : {acc*100:.2f}%")
    print(f"  Precision                       : {prec*100:.2f}%")
    print(f"  Recall                          : {rec*100:.2f}%")
    print(f"  F1 Score                        : {f1*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test_class, pred_class, labels=labels, zero_division=0))

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "Pred_Acc": pred_acc, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "pred_class": pred_class}

print("\nSTEP 7: Model results on the 20% TEST data")
lr_scores = evaluate("LINEAR REGRESSION", y_test, lr_pred)
rf_scores = evaluate("RANDOM FOREST", y_test, rf_pred)

# Summary table
summary = pd.DataFrame({
    "Linear Regression": [lr_scores[k] for k in ["MAE", "MSE", "RMSE", "R2", "Pred_Acc",
                                                  "Accuracy", "Precision", "Recall", "F1"]],
    "Random Forest":     [rf_scores[k] for k in ["MAE", "MSE", "RMSE", "R2", "Pred_Acc",
                                                  "Accuracy", "Precision", "Recall", "F1"]],
}, index=["MAE (°C)", "MSE", "RMSE (°C)", "R² Score", "Prediction Accuracy (±2°C)",
          "Accuracy", "Precision", "Recall", "F1 Score"])
print("\nFINAL COMPARISON TABLE")
print(summary.round(4))
summary.round(4).to_csv("model_comparison_results.csv")

best = "Random Forest" if rf_scores["R2"] > lr_scores["R2"] else "Linear Regression"
print(f"\nBEST MODEL: {best}")

# Linear Regression's Year coefficient = warming per year
print(f"\nLinear Regression says temperature rises by "
      f"{lr_model.coef_[0]*100:.2f} °C every 100 years (global warming trend).")

# -------------------------------------------------------------------
# STEP 8: VISUALISE RESULTS
# -------------------------------------------------------------------
# 8a. Actual vs Predicted for both models
n_sample = min(3000, len(y_test))
sample = np.random.RandomState(1).choice(len(y_test), n_sample, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, name, col in [(axes[0], lr_pred, "Linear Regression", "steelblue"),
                            (axes[1], rf_pred, "Random Forest", "seagreen")]:
    ax.scatter(y_test.values[sample], pred[sample], s=5, alpha=0.4, color=col)
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_title(f"{name}: Actual vs Predicted")
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.legend()
plt.tight_layout()
plt.savefig("3_actual_vs_predicted.png")
plt.show()

# 8b. Confusion matrices for both models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, scores, name, cmap in [(axes[0], lr_scores, "Linear Regression", "Blues"),
                               (axes[1], rf_scores, "Random Forest", "Greens")]:
    cm = confusion_matrix(y_test_class, scores["pred_class"], labels=labels)
    ax.imshow(cm, cmap=cmap)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix - {name}")
plt.tight_layout()
plt.savefig("4_confusion_matrix.png")
plt.show()

# 8c. Comparison bar chart of all percentage metrics
metric_names = ["R2", "Pred_Acc", "Accuracy", "Precision", "Recall", "F1"]
display_names = ["R² Score", "Pred. Acc\n(±2°C)", "Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(metric_names))
w = 0.38
lr_vals = [lr_scores[m] * 100 for m in metric_names]
rf_vals = [rf_scores[m] * 100 for m in metric_names]
plt.figure(figsize=(11, 5))
b1 = plt.bar(x - w / 2, lr_vals, w, label="Linear Regression", color="steelblue")
b2 = plt.bar(x + w / 2, rf_vals, w, label="Random Forest", color="seagreen")
for bars in (b1, b2):
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{b.get_height():.1f}", ha="center", fontsize=8)
plt.xticks(x, display_names)
plt.ylim(0, 110)
plt.ylabel("Score (%)")
plt.title("Model Comparison - All Metrics (higher is better)")
plt.legend()
plt.tight_layout()
plt.savefig("5_model_comparison.png")
plt.show()

# 8d. Error comparison (lower is better)
plt.figure(figsize=(7, 5))
err_names = ["MAE", "RMSE"]
x2 = np.arange(len(err_names))
plt.bar(x2 - w / 2, [lr_scores[m] for m in err_names], w, label="Linear Regression", color="steelblue")
plt.bar(x2 + w / 2, [rf_scores[m] for m in err_names], w, label="Random Forest", color="seagreen")
plt.xticks(x2, err_names)
plt.ylabel("Error (°C)")
plt.title("Model Errors (lower is better)")
plt.legend()
plt.tight_layout()
plt.savefig("6_error_comparison.png")
plt.show()

# 8e. Feature importance from Random Forest
plt.figure(figsize=(7, 4))
plt.barh(features, rf_model.feature_importances_, color="seagreen")
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("7_feature_importance.png")
plt.show()

# -------------------------------------------------------------------
# STEP 9: PREDICT FUTURE TEMPERATURE FOR A COUNTRY
# -------------------------------------------------------------------
country = "India"        # <-- change to any country in the dataset
if country in country_avg.index:
    code = df.loc[df["Country"] == country, "Country_Code"].iloc[0]
    print(f"\nSTEP 9: Predicted yearly average temperature for {country}")
    print(f"{'Year':<8}{'Linear Reg':>14}{'Random Forest':>16}")
    for year in [2020, 2030, 2050]:
        future = pd.DataFrame({"Year": [year] * 12,
                               "Month": range(1, 13),
                               "Country_Code": [code] * 12,
                               "Country_AvgTemp": [country_avg[country]] * 12})
        print(f"{year:<8}{lr_model.predict(future).mean():>12.2f}°C"
              f"{rf_model.predict(future).mean():>14.2f}°C")
    print("Note: Random Forest cannot extrapolate beyond 2013 (last year in data),")
    print("so Linear Regression is better for long-term FUTURE trend prediction,")
    print("while Random Forest is more accurate within the known time range.")

print("\nProject complete! Graphs (PNG) and model_comparison_results.csv")
print("are saved in the project folder.")
