import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv('data/heart.csv')
    
    print("Dataset loaded!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nAny missing values?")
    print(df.isnull().sum())
    print("\nTarget value counts:")
    print(df['target'].value_counts())
    
    # Features aur target alag karo
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train test split - 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nTraining set size:", X_train.shape)
    print("Testing set size:", X_test.shape)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    load_and_preprocess()
