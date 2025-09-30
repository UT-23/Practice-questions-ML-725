"""Experiment 3 — Linear regression predicting the digit label (treated as numeric)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

def main():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Convert regression predictions to integer labels for a simple accuracy check
    y_pred_round = np.rint(y_pred).astype(int)
    y_pred_round = np.clip(y_pred_round, 0, 9)
    acc = accuracy_score(y_test, y_pred_round)
    print("Accuracy after rounding predictions to nearest digit:", acc)

    print("Coefficients shape:", model.coef_.shape)

if __name__ == '__main__':
    main()
