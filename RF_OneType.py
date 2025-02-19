#!/usr/bin/env python3
"""
Random Forest Classification (one data type) with Grid Search and Cross-Validation
==================================================================

This script performs a Random Forest classification on either experimental or simulated data.
It conducts a grid search to find optimal hyperparameters and evaluates model performance
across multiple random splits to assess average accuracy and feature importances.

Author: Colin L. Williams
Date: 2024-02-19
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
import os


def load_data(path: str, features_file: str, labels_file: str, cutoff: int):
    """Load feature and label data with an option to exclude the first `cutoff` features."""
    try:
        X = np.loadtxt(os.path.join(path, features_file))[:, cutoff:]
        y = np.loadtxt(os.path.join(path, labels_file))
    except Exception as e:
        raise FileNotFoundError(f"Error loading files: {e}")
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features after cutoff of {cutoff}.")
    return X, y


def grid_search_rf(X_train, y_train, param_grid, seed):
    """Perform grid search to find the best Random Forest hyperparameters."""
    rf = RandomForestClassifier(random_state=seed)
    grid_search = GridSearchCV(rf, param_grid, cv=2, n_jobs=-1, verbose=1, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.cv_results_['mean_test_score']


def plot_feature_importances(mean_importances, std_importances):
    """Plot average feature importances with error bars."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mean_importances)), mean_importances, yerr=std_importances, capsize=5, color='skyblue')
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Importance")
    plt.title("Average Feature Importances with Error Bars")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


def main(args):
    # Load data
    X, y = load_data(args.path, args.features_file, args.labels_file, args.cutoff)

    # Hyperparameter grid for Random Forest
    param_grid = {
        'n_estimators': [5, 11, 13, 15, 17, 25, 53, 103],
        'max_depth': [2, 4, 5, 6, 7, 10, 12, 14],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3, 4]
    }

    training_acc, testing_acc, all_importances = [], [], np.zeros((args.iterations, X.shape[1]))

    for seed in range(args.iterations):
        print(f"\nIteration {seed + 1}/{args.iterations}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=seed, shuffle=True
        )

        # Grid search to find the best model
        best_model, cv_accuracies = grid_search_rf(X_train, y_train, param_grid, seed)
        print(f"Average CV Testing Accuracy: {np.mean(cv_accuracies) * 100:.2f}%")

        # Train and evaluate the best model
        best_model.fit(X_train, y_train)
        y_pred_train, y_pred_test = best_model.predict(X_train), best_model.predict(X_test)
        train_acc, test_acc = accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)
        
        training_acc.append(train_acc)
        testing_acc.append(test_acc)
        all_importances[seed, :] = best_model.feature_importances_

        print(f"Training Accuracy: {train_acc * 100:.2f}% | Testing Accuracy: {test_acc * 100:.2f}%")
        if seed == args.iterations - 1:  # Plot confusion matrix for final iteration
            plot_confusion_matrix(y_test, y_pred_test, title=f"Confusion Matrix (Seed {seed})")

    # Aggregate results
    mean_importances, std_importances = np.mean(all_importances, axis=0), np.std(all_importances, axis=0)
    print(f"\nFinal Average Training Accuracy: {np.mean(training_acc) * 100:.2f}% ± {np.std(training_acc) * 100:.2f}%")
    print(f"Final Average Testing Accuracy: {np.mean(testing_acc) * 100:.2f}% ± {np.std(testing_acc) * 100:.2f}%")

    plot_feature_importances(mean_importances, std_importances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest with Grid Search and Cross-Validation.")
    parser.add_argument("--path", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--features_file", type=str, required=True, help="Feature data file.")
    parser.add_argument("--labels_file", type=str, required=True, help="Labels data file.")
    parser.add_argument("--cutoff", type=int, default=0, help="Number of initial features to exclude.")
    parser.add_argument("--test_size", type=float, default=0.95, help="Proportion of data to use for testing.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of random seeds/splits.")

    args = parser.parse_args()
    main(args)
