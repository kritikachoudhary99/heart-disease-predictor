# Heart Disease Predictor - Model Training
import numpy as np  #numbers library 
from sklearn.linear_model import LogisticRegression #firstmodel
from sklearn.ensemble import RandomForestClassifier#second model
from sklearn.svm import SVC #third model
from sklearn.metrics import accuracy_score

def train_models(X_train, X_test, y_train, y_test): #(jo data model seekhega,jo data model pe test hoga, sahi ans "disease h ya nhi", testing ka sahi ans)
    # Model 1 - Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print("Logistic Regression Accuracy:", lr_acc)

    # Model 2 - Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print("Random Forest Accuracy:", rf_acc)

    # Model 3 - SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print("SVM Accuracy:", svm_acc)

    # Best model return karo
    models = {
        "Logistic Regression": lr_acc,
        "Random Forest": rf_acc,
        "SVM": svm_acc
    }
    best_model = max(models, key=models.get)
    print("\nBest Model:", best_model)
    
    return rf  # Random Forest save karenge

if __name__ == "__main__":
    from preprocess import load_and_preprocess
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    train_models(X_train, X_test, y_train, y_test)