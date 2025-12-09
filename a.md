# Regression

This time, we are doing an example on Regression. 

In machine learning, everything begins with data. Our firstt job is to load the dataset into the notebook so we can explore and prepare it.

https://www.kaggle.com/datasets/mirichoi0218/insurance


Regression is a supervised machine learning method used to predict continuous numerical values. **Examples:** predicting house prices, sales, insurance charges, temperature, etc. In contrast, classification predicts categories (yes/no, spam/not spam, etc.).





## Step 1
We open up colab and start like we did before:

```python
from google.colab import files
uploaded = files.upload()
```

We put our **insurance.csv** file there and carry on.

## Step 2
Now, we are going to load this file into a Pandas *dataframe*. We must understand the data. Before we go into training, we need to understand what we are working with.

* Pandas is the main library for working with tabular data.
* A *DataFrame* is like an Excel sheet inside Python.
* First step is to **load** and **looking at** the dataset.

```python
import pandas as pd
df = pd.read_csv("insurance.csv")
df.head()
```

Here, we read the data into a dataframe: `df`. All rows and columns are now stored in memory. Next, we display the first 5 rows (*df.head()*).

At this point, we verify if the data is loaded correctly. Also, it gives an initial feel of what kind of data we're dealing with.

At this point, we should check:
* What are the features?
* Which one is the **target** variable (the one we want to guess)?
* Are there both numeric and categorical columns?

## Step 3

We are now going to do basic data exploration. We need to know certain information about the dataset.

* How many rows and columns?
* What data types?
* Any missing values?
* What do the basic statistics look like?

* We use `df.shape` which returns `(rows, cols)`. This way we understand the size of the data.
* We use `df.info()`, which summarizes the data for us. We see the columns, number of data and their type.
* We use `df.describe()` which gives us *mean, std, min, max, quartiles*. This way, we can spot if there are outliers, strange data or skewed distributions.
* To check for missing values, we use `df.isnull().sum()`. This returns the number of missing values per column. For this dataset, we don't have any.


## Step 4: Understanding Feature Types & Preparing Data for Modeling

We see that some of the columns are numerical. We like numerical data in machine learning! However we also have *categorical* columns such as `sex`, `smoker` and `region`. 

We saw that ML models love numbers and want to work with them. Therefore, we need to turn these into numerical values! This is called **encoding**.

To see the datatypes, we type `df.dtypes`. As we see here, some columns are of `object` type, and some of them are numeric. We want to make all numeric. So, we are going to use **one-hot encoding**. After we prepare our data, we are going to apply encoding.

Remember that here, we need to have a target. And the target is: `charges`. Because we want to guess the *charges*. So now, we are going to have an X and Y. 

X being **all input columns** and Y being **the column we want to predict**.

```python
X = df.drop("charges", axis=1)
y = df["charges"]
```

So now, we drop *charges* from X because we don't want it, we want to guess it. 

Now we can apply encoding.

```python
X = pd.get_dummies(X, drop_first=True)
X.head()
```

Here, `get_dummies` is the one-hot encoding. We use *drop-first=true* because it helps us with the *dummy variable trap*. If you are using a linear model (like Linear Regression or Logistic Regression), the new columns will be perfectly correlated (you can derive one column's value from the others). This can cause issues. To solve this, you can drop one of the generated columns by adding drop_first=True.


---

Here we can do some visualization on the data.

```python
import matplotlib.pyplot as plt
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()
```

This creates a histogram. It shows us the distribution shape. For example if we check them;
we see:

* charges will show a right-skewed distribution (important for predicting!)
* bmi has often outliers (Very high bmi)
* age naturally uniform-ish

---

Correlation:

```python
df.corr(numeric_only=True)
#visual below
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```




## Step 5: Train/Test Split

When we train a machine learning model, we must evaluate it fairly.
If we train and test the model on the same data, we get:

* artificially high accuracy
* a model that memorizes instead of generalizing
* poor performance on real-world data


So we divide the dataset into a **training set** and **testing set**. Training set is used to train/fit the model and usually accounts for 70-80\% of data. 

Testing set, consists of data that the model has **never seen**. This data is only used for evaluation. It lets us calculate how well the model generalizes. 

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

> **Important**: The test set must NEVER be used for training. It must stay “unseen” until final evaluation. This prevents overfitting and ensures realistic model performance.

## Step 6: Training Our First Regression Model

Linear Regression is one of the oldest and most interpretable machine learning models.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

Here, we train the model using `LinearRegression()`. The model mathematically adjusts its coefficients until it finds the best prediction rule for the training data.

At this point, we **trained** the data but we did not **test** it yet. So now, we are going to **evalute** how well our model works.

We trained a Linear Regression model using `X_train` and `Y_train`. Now we are going to evalute the model using `X_test` and `Y_test`.

```python
y_pred = model.predict(X_test)
```

This makes predictions on test data. And it did! 

Now we need to compute our evaluation metrics. In regression, the standard evaluation metrics are:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* $R^{2}$ score (Coefficient of Determination)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

mae, mse, rmse, r2
```

* MAE: On average, we are off by X dollars
* MSE: Penalizes large errors
* $R^{2}$ score: Ranges from 0 to 1 (sometimes negative), Closer to 1 is better. Measures how much of the variance the model explains

| Metric | Meaning                      |
| ------ | ---------------------------- |
| MAE    | average prediction error     |
| RMSE   | how wrong the model is, in $ |
| R²     | model quality                |


| Metric   | Your Value     | Meaning                                      |
| -------- | -------------- | -------------------------------------------- |
| **MAE**  | **4181.19**    | On average, predictions are off by $4181     |
| **MSE**  | **33,596,915** | Squared error (harder to interpret directly) |
| **RMSE** | **5796.28**    | Typical prediction error ≈ $5800             |
| **R²**   | **0.784**      | Model explains **78.4%** of the variance     |


* $R^2$ value is good at 0.784. 
* RMSE: On average, the model’s predictions are about $5800 away from the real value
* MAE: Typically, we are off by around $4180
  
> We built a regression model that explains 78% of the variation in insurance charges, and its predictions are typically within $4,000–$5,800 of the true cost


## Step 8: 

| **Regression Type**                                 | **Best For**                                   | **Strengths**                             | **Weaknesses**                                        |
| --------------------------------------------------- | ---------------------------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| **Linear Regression**                               | Data with roughly linear relationships         | Simple, fast, interpretable               | Cannot model curves or complex patterns               |
| **Polynomial Regression**                           | Curved relationships                           | Captures nonlinearity, easy to implement  | Overfits easily, poor extrapolation                   |
| **Ridge Regression (L2)**                           | Data with multicollinearity                    | Reduces overfitting, stable               | Keeps all features, no feature elimination            |
| **Lasso Regression (L1)**                           | When feature selection is needed               | Eliminates irrelevant features            | Can behave unpredictably when features are correlated |
| **ElasticNet**                                      | Combination of L1 + L2 needs                   | Balanced feature shrinking + selection    | Requires tuning 2 hyperparameters                     |
| **Decision Tree Regression**                        | Data with nonlinear relationships              | Easy to visualize, interpretable          | Overfits without pruning                              |
| **Random Forest Regression**                        | Most real-world tabular data                   | Very strong performance, robust           | Less interpretable, slower                            |
| **Gradient Boosting (XGBoost, LightGBM, CatBoost)** | High accuracy on complex datasets              | State-of-the-art performance              | Can overfit if poorly tuned                           |
| **Support Vector Regression (SVR)**                 | Small to medium datasets, nonlinear patterns   | Good with kernels, handles complex shapes | Slow on large datasets, sensitive to parameters       |
| **KNN Regression**                                  | Smooth relationships, low-dimensional data     | Simple, no training step                  | Slow prediction, poor with many features              |
| **Neural Network Regression (MLP)**                 | Medium to large datasets with complex patterns | Very flexible                             | Requires lots of data, harder to interpret            |
| **CNN Regression**                                  | Image-based regression tasks                   | Extracts spatial patterns                 | Needs large datasets, GPU                             |
| **RNN/LSTM Regression**                             | Time-series forecasting                        | Remembers sequences, trends               | Harder to train, risk of vanishing gradients          |
| **Transformer Regression**                          | Advanced time-series & sequence modeling       | State-of-the-art performance              | Resource-heavy, advanced topic                        |


---

Now, the easy way to run all these regression models are as follows:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
```

```python
df = pd.read_csv("insurance.csv")

# Separate input and target
X = df.drop("charges", axis=1)
y = df["charges"]

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Since SVR & KNN need **scaled** data, we add this part:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
``` 

```python
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(kernel='rbf')
}
```

```python
def evaluate_model(name, model, Xtr, Xte):
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return [name, mae, rmse, r2]
```

```python
results = []

for name, model in models.items():
    # scaled versions for models that require it
    if name in ["KNN", "SVR"]:
        res = evaluate_model(name, model, X_train_scaled, X_test_scaled)
    else:
        res = evaluate_model(name, model, X_train, X_test)
    
    results.append(res)
```

```python
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R²"]
)

results_df
results_df.sort_values(by="R²", ascending=False)
```

At this point, we can see that Random Forest works best!


## Step 10

Let's add some visualizations. 

1. Predicted vs. Actual plot (how close are predictions?)
2. Feature Importance plot (which features matter most?)

Since in the last example we ran everything in a loop, we now need to run random forest again so that the following examples now about it.

```python
# Re-train Random Forest separately for visualization
best_model = RandomForestRegressor(n_estimators=200, random_state=42)
best_model.fit(X_train, y_train)

# Predictions for visualization
y_pred_best = best_model.predict(X_test)
```

Now we can create plot 1: predicted vs actual

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```

And we can do **feature importance plot**.

```python
import numpy as np

importances = best_model.feature_importances_
features = X.columns

indices = np.argsort(importances)

plt.figure(figsize=(10,8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()
```

---

### residual plot

Residuals = Actual – Predicted

A good model should have residuals centered around 0 with no obvious pattern.

```python
residuals = y_test - y_pred_best
plt.figure(figsize=(8,6))
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot (Random Forest)")
plt.show()
```

```
If points are randomly scattered → good

If there is a curved shape → linear model would fail

If there is a funnel shape → variance isn't constant (heteroscedasticity)
```

### Histogram of Prediction Errors

This shows how large the mistakes are.

```python
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.show()
```

```
Centered around 0 → good

Long right tail → model underestimates high charges

Long left tail → model overestimates high charges
```

---

### SHAP

```python
!pip install shap
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
```

```
How each feature influences each prediction

Global and local interpretability

Why smoking is the dominant predictor in this dataset
```




