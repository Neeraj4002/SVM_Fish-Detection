# core.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import matplotlib.pyplot as plt


def load_and_process(file):
    """
    Load Excel dataset, parse datetime, create binary disease label, and return df, features X and target y.
    """
    df = pd.read_excel(file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df['disease'] = (df['Disease Occurrence (Cases)'] > 1.5).astype(int)
    features = ['Dissolved Oxygen (mg/L)', 'pH']
    X = df[features]
    y = df['disease']
    return df, X, y


def split_train_test(X, y, window_size, step):
    """
    Find a contiguous test window of given size containing both classes, then split train/test accordingly.
    """
    test_start = test_end = None
    for i in range(0, len(y) - window_size, step):
        chunk = y.iloc[i:i+window_size]
        if chunk.nunique() == 2:
            test_start, test_end = i, i+window_size
            break
    if test_start is None:
        raise ValueError("Couldn't find a valid test window with both classes.")
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]
    X_train = pd.concat([X.iloc[:test_start], X.iloc[test_end:]])
    y_train = pd.concat([y.iloc[:test_start], y.iloc[test_end:]])
    return X_train, X_test, y_train, y_test


def run_grid_search(X_train, y_train, param_grid):
    """
    Build a pipeline (SelectKBest -> StandardScaler -> SVC), run GridSearchCV, and return the fitted GridSearchCV.
    """
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(f_classif, k='all')),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(model, X_test, y_test):
    """
    Predict on test set and compute accuracy, precision, recall, F1, classification report, confusion matrix.
    """
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def plot_confusion_matrix(cm):
    """
    Return a matplotlib Figure of the confusion matrix heatmap.
    """
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy','Disease'], yticklabels=['Healthy','Disease'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig


def plot_decision_boundary(model, X_train, y_train, X_test, y_test):
    """
    Return a matplotlib Figure showing decision boundary overlaid with train/test points.
    """
    fig, ax = plt.subplots()
    x_min, x_max = X_train.iloc[:,0].min() - 0.5, X_train.iloc[:,0].max() + 0.5
    y_min, y_max = X_train.iloc[:,1].min() - 0.5, X_train.iloc[:,1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train, edgecolors='k', alpha=0.7, label='Train')
    ax.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=y_test, edgecolors='k', marker='s', alpha=1.0, label='Test')
    ax.set_xlabel('Dissolved Oxygen (mg/L)')
    ax.set_ylabel('pH')
    ax.legend()
    return fig
import joblib
import os

def save_model(model, path='model.pkl'):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, path)

def load_model(path='model.pkl'):
    """
    Load a trained model from disk.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError("Saved model not found.")
