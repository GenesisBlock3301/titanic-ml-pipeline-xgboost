

## 🎯 Blog Title:

**A Complete Machine Learning Project — From Data Preprocessing to Model Validation**

---

## 📌 Introduction

I recently completed Kaggle’s Intermediate Machine Learning course. Throughout this course, I learned about handling Missing Values, Categorical Variables, Pipelines, Cross-Validation, XGBoost, and Data Leakage. To solidify these concepts, I built a small project that puts these ideas into practice.

In this blog, I’ll walk you through how to build a complete machine learning project — from preprocessing the dataset to validating the model.

---

## 🧠 Step 1: Identify Feature Types

First, we need to identify which columns are **numerical** and which are **categorical**, because the preprocessing steps for these types are different.

```python
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
```

---

## 🧹 Step 2: Handle Missing Values

Missing values can negatively affect model training. So, we impute missing values using the **median** for numerical columns and the **most frequent value** for categorical ones.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

---

## 🔗 Step 3: Build the Preprocessing Pipeline

We use `Pipeline` and `ColumnTransformer` to combine preprocessing steps and make the workflow cleaner and more reproducible.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

---

## 🧠 Step 4: Create the Model and Combine with Pipeline

I used `XGBClassifier` for this project and integrated it with the preprocessing pipeline.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

from sklearn.pipeline import Pipeline

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])
```

---

## 📊 Step 5: Validate the Model Using Cross-Validation

To evaluate the model’s performance, I used `cross_val_score`, which splits the data into 5 folds and returns the average accuracy.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Average accuracy: {:.2f}%".format(scores.mean() * 100))
```

---

## 📈 Step 6 (Optional): Visualize the Scores

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(scores)
plt.title("Cross-Validation Accuracy Scores")
plt.show()
```

---

## ✅ Key Takeaways

* **Pipelines** help make the code clean, reusable, and less prone to data leakage.
* **Cross-validation** gives a more reliable estimation of model performance.
* **XGBoost** is a powerful model that performs well even with default settings.
* **Data preprocessing** is crucial — even the best model won’t work well on poorly prepared data.

---

## 📁 Project Repository

GitHub Repo: [https://github.com/GenesisBlock3301/titanic-ml-pipeline-xgboost.git](https://github.com/GenesisBlock3301/titanic-ml-pipeline-xgboost.git)
(👉 Replace this with your actual link if needed.)

---

## 🧑‍💻 If You’re Just Getting Started...

This project and blog post can be a great resource for beginners who are learning machine learning. If you have any questions or suggestions — feel free to drop a message or comment! 😊

---

## 🔖 Hashtags

```
#MachineLearning #Kaggle #DataScience #XGBoost #MLPipeline #Python #CrossValidation #MLProject
```
