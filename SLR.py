import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

data = pd.read_csv('vo2max_data.csv')

print("Dataset Head:")
print(data.head())

print("\nDataset Description:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

X = data.drop(columns=['VO2Max'])
y = data['VO2Max']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(score_func=f_regression, k='all')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print("\nSelected Features:")
print(selected_features)

lr_model = LinearRegression()
lr_model.fit(X_train_selected, y_train)

y_pred_lr = lr_model.predict(X_test_selected)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Mean Squared Error: {mse_lr:.2f}")
print(f"Linear Regression R-squared: {r2_lr:.2f}")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)

y_pred_rf = rf_model.predict(X_test_selected)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\nRandom Forest Mean Squared Error: {mse_rf:.2f}")
print(f"Random Forest R-squared: {r2_rf:.2f}")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_selected, y_train)

best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_selected)

mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

print(f"\nTuned Random Forest Mean Squared Error: {mse_best_rf:.2f}")
print(f"Tuned Random Forest R-squared: {r2_best_rf:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_rf, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Actual vs Predicted VO2 Max (Tuned Random Forest)')
plt.xlabel('Actual VO2 Max')
plt.ylabel('Predicted VO2 Max')
plt.grid()
plt.show()

cross_val_scores = cross_val_score(best_rf_model, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = np.mean(np.abs(cross_val_scores))

print(f"\nCross-Validation Mean Squared Error: {mean_cv_score:.2f}")
