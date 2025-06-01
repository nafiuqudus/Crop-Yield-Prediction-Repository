# 🌾 Crop Yield Prediction System

This project is a **machine learning-based web application** built to help farmers, researchers, and policymakers **predict crop yield** using key environmental and input parameters such as **rainfall**, **pesticide usage**, **temperature**, **area**, and **crop type**. The system supports predictions across various countries and crop types, using historical data sourced from [Kaggle](https://www.kaggle.com/).

---

## Project Highlights

- ✅ Dataset sourced from Kaggle
- ✅ Feature engineering using `ColumnTransformer`
- ✅ Multiple regression models tested
- ✅ Best model selected: `DecisionTreeRegressor`
- ✅ Web application built for real-time predictions
- ✅ Saved trained model and preprocessor using `pickle`
- ✅ Easily deployable and user-friendly interface

---

## 🧪 Features Used

- `Year`
- `Average Rainfall (mm/year)`
- `Pesticides Used (tonnes)`
- `Average Temperature (°C)`
- `Area (ha)`
- `Item (Crop Type)`

---

## 🧠 ML Workflow Summary

```python
# Features and target variable
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing: Scaling + One-hot encoding
preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0,1,2,3]),
        ('OneHotEncode', OneHotEncoder(drop='first'), [4,5])
    ],
    remainder='passthrough'
)

# Transform data
X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)

# Train and evaluate models
models = {
    'linear regression' : LinearRegression(),
    'lasso' : Lasso(),
    'ridge' : Ridge(),
    'KNN' : KNeighborsRegressor(),
    'dtr' : DecisionTreeRegressor()
}

for name, md in models.items():
    md.fit(X_train_dummy, y_train)
    y_predict = md.predict(X_test_dummy)
    print(f"{name}: MAE = {mean_absolute_error(y_test, y_predict)}, R² = {r2_score(y_test, y_predict)}")

#Final Model
The best-performing model was Decision Tree Regressor, which was used in the final deployment. The model and preprocessor were serialized using pickle:

pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocesser, open('preprocesser.pkl', 'wb'))

#Predictive System Function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
    transform_features = preprocesser.transform(features)
    predictive_yield = dtr.predict(transform_features).reshape(-1,1)
    return predictive_yield[0][0]
