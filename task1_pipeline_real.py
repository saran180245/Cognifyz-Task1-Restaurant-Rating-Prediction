import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------- 1. LOAD DATA ----------
df =pd.read_csv("cognifyz_restaurant_dataset.csv")
print("Original shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ---------- 2. PREPROCESSING ----------

# Handle missing values (Cuisines has a few nulls)
df['Cuisines'] = df['Cuisines'].fillna('Not Specified')

# Drop columns that aren't useful predictors / are identifiers or redundant with target
drop_cols = ['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose',
             'Rating color', 'Rating text', 'Switch to order menu', 'Currency']
df = df.drop(columns=drop_cols)

# Encode categorical variables
cat_cols = ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 'Is delivering now']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

feature_cols = ['Country Code', 'City_enc', 'Longitude', 'Latitude', 'Cuisines_enc',
                'Average Cost for two', 'Currency' if False else 'Price range',
                'Has Table booking_enc', 'Has Online delivery_enc', 'Is delivering now_enc', 'Votes']

X = df[feature_cols]
y = df['Aggregate rating']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------- 3. MODEL TRAINING ----------
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest Regression': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2, 'model': model}
    print(f"\n{name}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2 : {r2:.4f}")

# ---------- 4. FEATURE IMPORTANCE (best model: Random Forest) ----------
best_model = results['Random Forest Regression']['model']
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances)

# Save summary results
summary_df = pd.DataFrame({name: {'MSE': r['MSE'], 'MAE': r['MAE'], 'R2': r['R2']} for name, r in results.items()}).T
summary_df.to_csv('model_comparison_results_real.csv')
importances.to_csv('feature_importance_real.csv', header=['Importance'])

print("\nSaved: model_comparison_results_real.csv, feature_importance_real.csv")
