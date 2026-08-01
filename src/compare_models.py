import joblib

from load_data import load_data

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score


# -------------------------
# Load Dataset
# -------------------------

df = load_data()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# -------------------------
# Train / Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Models
# -------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# -------------------------
# Compare Models
# -------------------------

results = []

best_trained_model = None
best_model_name = ""
best_mse = None
best_r2 = float("-inf")


print("=" * 65)
print(f"{'Model':<20} {'MSE':<15} {'R2 Score'}")
print("=" * 65)


for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((name, mse, r2))

    print(f"{name:<20} {mse:<15.4f} {r2:.4f}")

    # Save Best Model
    if r2 > best_r2:
        best_r2 = r2
        best_mse = mse
        best_model_name = name
        best_trained_model = model


# -------------------------
# Save Best Model
# -------------------------

joblib.dump(best_trained_model, "models/best_model.pkl")


# -------------------------
# Final Result
# -------------------------

print("\n" + "=" * 65)
print("Best Model")
print("=" * 65)

print(f"Name : {best_model_name}")
print(f"R2   : {best_r2:.4f}")
print(f"MSE  : {best_mse:.4f}")

print("\nBest model saved successfully!")
print("Location: models/best_model.pkl")