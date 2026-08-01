import joblib
import matplotlib.pyplot as plt
import pandas as pd

# بارگذاری مدل ذخیره‌شده
model = joblib.load("models/best_model.pkl")

# نام ویژگی‌ها
feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

# گرفتن اهمیت ویژگی‌ها
importance = model.feature_importances_

# ساخت DataFrame
df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# مرتب‌سازی
df = df.sort_values(by="Importance", ascending=False)

# چاپ جدول
print(df)

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.bar(df["Feature"], df["Importance"])

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

# ذخیره نمودار
plt.savefig("images/feature_importance.png")

# نمایش نمودار
plt.show()