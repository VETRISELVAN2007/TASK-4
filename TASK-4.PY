import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset

data = pd.read_csv("advertising.csv")

print("Dataset Loaded Successfully\n")

print("First 5 Records")

print(data.head())

print("\nDataset Shape:", data.shape)

# Visualization

sns.pairplot(data)

plt.show()

plt.figure(figsize=(6,5))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.show()

# Features and target

X = data[["TV", "Radio", "Newspaper"]]

y = data["Sales"]

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

model = LinearRegression()

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print(f"\nR² Score: {r2:.2f}")

print(f"Mean Absolute Error: {mae:.2f}")

# Actual vs Predicted graph

plt.figure(figsize=(7,5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()]
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# Feature importance

importance = pd.Series(
    model.coef_,
    index=X.columns
)

plt.figure(figsize=(6,4))

importance.plot(kind="bar")

plt.title("Advertising Impact on Sales")

plt.ylabel("Impact Score")

plt.show()

# User prediction

print("\nEnter Advertising Budget")

try:

    tv = float(input("TV Budget: "))

    radio = float(input("Radio Budget: "))

    newspaper = float(input("Newspaper Budget: "))

    sample = np.array([
        [
            tv,
            radio,
            newspaper
        ]
    ])

    sales = model.predict(sample)

    print(f"\nPredicted Sales: {sales[0]:.2f}")

except ValueError:

    print("Enter numeric values only.")