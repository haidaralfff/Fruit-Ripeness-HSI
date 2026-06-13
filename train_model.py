import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os
import pickle
import time
# pyrefly: ignore [missing-import]
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns

def main():
    train_csv = os.path.join('data', 'processed', 'train_features.csv')
    test_csv = os.path.join('data', 'processed', 'test_features.csv')

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    print("Training data shape:", df_train.shape)
    print("Testing data shape:", df_test.shape)

    X_train = df_train.drop(['fruit', 'label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['fruit', 'label'], axis=1)
    y_test = df_test['label']

    print("Training model...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    print("Evaluating model...")
    start_time = time.time()
    y_pred = rf.predict(X_test)
    end_time = time.time()

    latency_per_image = (end_time - start_time) / len(X_test) * 1000
    print(f'\nInference Latency: {latency_per_image:.4f} ms per image')

    print('\nAccuracy:', accuracy_score(y_test, y_pred))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'confusion_matrix.png'))
    plt.close()

    # Feature Importance
    importances = rf.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)
    plt.figure(figsize=(8,6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'feature_importance.png'))
    plt.close()

    print("Saving model to models/random_forest.pkl...")
    with open(os.path.join('models', 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf, f)

    print("Model saved successfully!")

if __name__ == "__main__":
    main()
