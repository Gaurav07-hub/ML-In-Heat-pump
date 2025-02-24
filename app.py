import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset (or replace with real data)
np.random.seed(42)
data_size = 1000
evaporator_temp = np.random.uniform(-5, 15, data_size)  # in °C
condenser_temp = np.random.uniform(30, 50, data_size)   # in °C
power_input = np.random.uniform(1, 5, data_size)        # in kW
heat_output = power_input * np.random.uniform(3, 4, data_size)  # in kW
cop_actual = heat_output / power_input  # COP calculation

# Create DataFrame
df = pd.DataFrame({
    'Evaporator_Temperature': evaporator_temp,
    'Condenser_Temperature': condenser_temp,
    'Power_Input': power_input,
    'Heat_Output': heat_output,
    'COP_Actual': cop_actual
})

# Split data
X = df[['Evaporator_Temperature', 'Condenser_Temperature', 'Power_Input']]
y = df['COP_Actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name} -> MSE: {mse:.3f}, R²: {r2:.3f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Plot Performance Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df["R2"])
plt.title("R² Score Comparison of Models")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.show()