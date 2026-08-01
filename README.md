# 🏠 Housing Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts California housing prices using Machine Learning models.

The project demonstrates a complete Machine Learning workflow, including:

* Data loading
* Exploratory Data Analysis (EDA)
* Data visualization
* Model training
* Model comparison
* Model evaluation
* Model saving
* House price prediction
* Feature importance analysis

---

## 📂 Project Structure

```text
housing-price-prediction/
│
├── data/
│   └── california_housing.csv
│
├── images/
│   ├── histograms.png
│   └── feature_importance.png
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│
├── src/
│   ├── load_data.py
│   ├── explore_data.py
│   ├── visualize.py
│   ├── compare_models.py
│   ├── feature_importance.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset: California Housing Dataset

Features:

* Median Income
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

Target:

* Median House Value

---

## 🤖 Machine Learning Models

The following regression models were trained and compared:

| Model             |        MSE |   R² Score |
| ----------------- | ---------: | ---------: |
| Linear Regression |     0.5559 |     0.5758 |
| Decision Tree     |     0.4943 |     0.6228 |
| Random Forest     | **0.2560** | **0.8046** |

### ✅ Best Model

Random Forest Regressor

R² Score: **0.8046**

---

## 📈 Feature Importance

The Random Forest model identified the following features as the most important:

| Feature    | Importance |
| ---------- | ---------: |
| MedInc     |      0.525 |
| AveOccup   |      0.138 |
| Latitude   |      0.089 |
| Longitude  |      0.088 |
| HouseAge   |      0.055 |
| AveRooms   |      0.044 |
| Population |      0.031 |
| AveBedrms  |      0.030 |

Feature importance chart:

```
images/feature_importance.png
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd housing-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Compare all models:

```bash
python src/compare_models.py
```

Predict a new house price:

```bash
python src/predict.py
```

Generate feature importance chart:

```bash
python src/feature_importance.py
```

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

## 📌 Results

* Built three regression models.
* Compared their performance.
* Selected the best-performing model.
* Saved the trained model.
* Implemented a prediction script.
* Visualized feature importance.

---

## 👨‍💻 Author

Developed as a Machine Learning portfolio project.
