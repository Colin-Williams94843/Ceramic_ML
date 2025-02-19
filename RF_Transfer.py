#!/usr/bin/env python3
"""
Random Forest Classification (Transfer Learning)
==================================================================

This script performs a Random Forest classification with transfer learning, that is, a portion
of experimental data (exp_train_pct) combined with all the simulated data in the training set 
for evaluation of unseen experimental data.

It conducts a grid search to find optimal hyperparameters and evaluate model performance
across multiple random splits to assess average accuracy and feature importance.

Author: Colin L. Williams
Date: 2024-02-19
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import argparse

# Main function to execute the model training and evaluation
def main(args):
    # Load and Prepare the Data
    print(f"Loading data from path: {args.path}")
    
    X_simulated = np.loadtxt(f"{args.path}All_T1T2_Sim_Freq_300.txt")  # Frequency list for simulated data
    X_simulated = X_simulated[:, args.cutoff:]  # Only second half of frequency range
    y_simulated = np.loadtxt(f'{args.path}T1T2_SimLabels_300.txt')  # Labels for simulated data

    X_experimental = np.loadtxt(f"{args.path}Experiments_65_99TallestPeaks.txt")  # Frequency list for experimental data
    X_experimental = X_experimental[:, args.cutoff:]
    y_experimental = np.loadtxt(f"{args.path}Experiments_65_Labels.txt")  # Labels for experimental data

    print(f"Simulated training data: {X_simulated.shape}")
    print(f"Experimental data: {X_experimental.shape}\n")

    # Initialize arrays for storing confusion matrices and importances
    all_importances = np.zeros((args.iterations, X_simulated.shape[1]))  # Match feature count
    all_confusion_matrices = []

    accuracies_test = []
    accuracies_train = []

    # Loop through multiple seeds for stability testing
    for seed in range(args.iterations):  # Loop of n random seeds to test stability
        # Split experimental data into training and testing subsets
        X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
            X_experimental, y_experimental, test_size=(1 - args.exp_train_pct), random_state=seed, stratify=y_experimental
        )

        # Combine simulated training data with the selected experimental training data
        X_combined_train = np.vstack((X_simulated, X_exp_train))
        y_combined_train = np.hstack((y_simulated, y_exp_train))

        # Set up Grid Search for Random Forest
        param_grid = {
            'n_estimators': [5, 11, 13, 15, 17, 25, 53, 103],  # Number of trees in the forest
            'max_depth': [2, 4, 5, 6, 7, 10, 12, 14],  # Maximum depth of trees
            'min_samples_split': [2, 3, 5],  # Minimum samples required to split an internal node
            'min_samples_leaf': [1, 2, 3, 4]  # Minimum samples required to be at a leaf node
        }

        rf = RandomForestClassifier(random_state=seed)

        # Grid search with cross-validation
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
        grid_search.fit(X_combined_train, y_combined_train)

        # Get the best model from grid search
        best_training_model = grid_search.best_estimator_

        print(f"\nBest model parameters from grid search: {grid_search.best_params_}\n")

        # Evaluate the best model on the combined training data
        y_pred_train = best_training_model.predict(X_combined_train)
        accuracy_train = accuracy_score(y_combined_train, y_pred_train)

        # Store and print the training accuracy for this seed
        accuracies_train.append(accuracy_train)
        print(f"Training accuracy for seed {seed}: {accuracy_train * 100:.2f}%\n")

        # Evaluate the trained model on the experimental test set
        y_pred_exp = best_training_model.predict(X_exp_test)
        accuracy_exp = accuracy_score(y_exp_test, y_pred_exp)

        # Calculate confusion matrix for this iteration
        cm = confusion_matrix(y_exp_test, y_pred_exp)
        all_confusion_matrices.append(cm)

        # Store and print the testing accuracy for this seed
        accuracies_test.append(accuracy_exp)
        print(f"Testing accuracy for seed {seed}: {accuracy_exp * 100:.2f}%\n")

        # Get feature importances
        importances = best_training_model.feature_importances_
        all_importances[seed, :] = importances  # Store importances for this iteration

    # Compute mean and std of importances across iterations
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    # Calculate the average training and testing accuracy across all seeds
    average_train_accuracy = np.mean(accuracies_train)
    train_std = np.std(accuracies_train)

    average_test_accuracy = np.mean(accuracies_test)
    test_std = np.std(accuracies_test)

    print(f"\nAverage training accuracy: {average_train_accuracy * 100:.2f}% ± {train_std * 100:.2f}%\n")
    print(f"Average testing accuracy: {average_test_accuracy * 100:.2f}% ± {test_std * 100:.2f}%\n")

    # Plot confusion matrix
    average_cm = np.mean(all_confusion_matrices, axis=0)
    std_cm = np.std(all_confusion_matrices, axis=0)

    new_labels = [1, 2]  # New labels for your classes
    fig, ax = plt.subplots(figsize=(5, 5))  # Increase figure size
    disp = ConfusionMatrixDisplay(confusion_matrix=average_cm, display_labels=new_labels)
    disp.plot(ax=ax, cmap='Blues')
    
    # Remove any existing text annotations if they exist
    for text in ax.texts:
    	text.set_visible(False)

    # Add custom text annotations with larger, bold font
    for i, (row, row_std) in enumerate(zip(average_cm, std_cm)):
        for j, (val, val_std) in enumerate(zip(row, row_std)):
            text = ax.text(j, i, f"{val:.2f}\n±{val_std:.2f}",
                           ha="center", va="center",
                           fontsize=14,  # Increase font size
                           fontweight='bold',  # Make font bold
                           color="white" if val > average_cm.max() / 2 else "black")

    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    plt.title('Average Confusion Matrix', fontsize=18)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(new_labels, fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(new_labels, fontsize=12)

    plt.tight_layout()
    plt.show()

    # Plot the average importances with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mean_importances)), mean_importances, yerr=std_importances, capsize=5, color='skyblue')
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Importance")
    plt.title("Average Feature Importances with Error Bars")
    plt.tight_layout()
    plt.show()

# Parse arguments for flexible execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Analysis on Simulated and Experimental Data")
    parser.add_argument('--path', type=str, required=True, help="Path to the data files")
    parser.add_argument('--cutoff', type=int, default=0, help="Number of initial features to exclude")
    parser.add_argument('--iterations', type=int, default=100, help="Number of iterations for stability testing")
    parser.add_argument('--exp_train_pct', type=float, default=0.04, help="Percentage of experimental data for training")

    args = parser.parse_args()
    main(args)
