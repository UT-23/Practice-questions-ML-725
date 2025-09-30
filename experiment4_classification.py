"""Experiment 4 — Gaussian Naive Bayes (Bayesian classifier) and SVM on digits."""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gaussian Naive Bayes (a simple Bayesian classifier)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_nb = nb.predict(X_test)
    print("Gaussian NB accuracy:", accuracy_score(y_test, y_nb))
    print("\nClassification report (NB):\n", classification_report(y_test, y_nb))

    # Support Vector Machine
    svm = SVC()
    svm.fit(X_train, y_train)
    y_svm = svm.predict(X_test)
    print("SVM accuracy:", accuracy_score(y_test, y_svm))
    print("\nClassification report (SVM):\n", classification_report(y_test, y_svm))

    print("Confusion matrix (SVM):\n", confusion_matrix(y_test, y_svm))

if __name__ == '__main__':
    main()
