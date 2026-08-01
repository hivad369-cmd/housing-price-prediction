import joblib
import pandas as pd

# بارگذاری مدل
model = joblib.load("models/best_model.pkl")

print("=" * 50)
print("House Price Prediction")
print("=" * 50)

# گرفتن اطلاعات از کاربر
med_inc = float(input("Median Income: "))
house_age = float(input("House Age: "))
ave_rooms = float(input("Average Rooms: "))
ave_bedrooms = float(input("Average Bedrooms: "))
population = float(input("Population: "))
ave_occup = float(input("Average Occupancy: "))
latitude = float(input("Latitude: "))
longitude = float(input("Longitude: "))

# ساخت DataFrame
new_data = pd.DataFrame([{
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrooms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude
}])

# پیش‌بینی
prediction = model.predict(new_data)

print("\n" + "=" * 50)
print(f"Predicted House Value: {prediction[0]:.3f}")
print("=" * 50)