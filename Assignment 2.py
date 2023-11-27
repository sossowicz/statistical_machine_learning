"""
COMP SCI 7314 Introduction to Statistical Machine Learning || Assignment 2
Submitted By: Arpit Gole || a1814270

Below program demonstrates working of Support Vector Machines (SVM) in the
following formats:
1. Soft-margin Linear SVM in it's primal form.
2. Soft-margin Linear SVM in it's dual form.
3. SVM implemented using Scikit-Learn.
"""

# Importing Packages
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def load_data(file_name):
    """
    Loads the file and sets appropriate label value.
    :param file_name: Path of the file.
    :return: n-d array in the form of Features and labels.
    """
    # Reading csv file
    df = pd.read_csv(file_name, header=None)

    # Changing the label values from 0 to -1
    # Keeping it consistent with the actual definition of SVM
    df.loc[df[0] == 0, 0] = -1

    # Separating features and labels
    return df.drop([0], axis=1).to_numpy(), df[0].to_numpy()


def svm_train_primal(data_train, label_train, regularisation_para_c):
    """
    Train a SVM classifier using primal formulation
    :param data_train: Training features for training the model
    :param label_train: Training class label relative to training features
    :param regularisation_para_c: Regularization parameter
    :return: Trained SVM model
    """
    # Fetching the no. of the data points and no. of features.
    n, f = np.shape(data_train)

    # Define Variables
    w = cp.Variable(shape=(f, 1), name='w', var_id=1)
    psi = cp.Variable(shape=(n, 1), name='psi', var_id=2)
    b = cp.Variable(name='b', var_id=3)
    c = regularisation_para_c

    # Objective to be solved
    objective = cp.Minimize(0.5 * cp.square(cp.norm(w, 2)) + c / n * cp.sum(psi))

    # Constraints in the problems
    constraints = [cp.multiply(label_train.reshape(-1, 1), data_train @ w + b) >= 1 - psi, psi >= 0]

    # Define and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=debug_mode)

    # Return the trained model
    return prob


def svm_predict_primal(data_test, label_test, svm_model):
    """
    Calculate the accuracy of the trained SVM model using primal formulation
    :param data_test: Testing features to obtain predictions for
    :param label_test: Testing class labels relative to the testing features
    :param svm_model: Trained SVM model using primal formulation
    :return: Accuracy on the test data
    """
    # Fetch the trained parameters
    w = svm_model.var_dict['w'].value
    b = svm_model.var_dict['b'].value

    # Obtain the predictions for test data
    y_predicted = data_test @ w + b

    # Compute the predicted class
    y_predicted[y_predicted > 0] = 1
    y_predicted[y_predicted <= 0] = -1

    # Calculate the predicted accuracy
    test_accuracy = accuracy_score(label_test.reshape(-1, 1), y_predicted)

    return test_accuracy


def svm_train_dual(data_train, label_train, regularisation_para_c):
    """
    Train a SVM classifier using dual formulation
    :param data_train: Training features for training the model
    :param label_train: Training class label relative to training features
    :param regularisation_para_c: Regularization parameter
    :return: Trained SVM model
    """
    # Fetching the no. of the data points and no. of features.
    n, f = np.shape(data_train)

    # Define variables
    alpha = cp.Variable(n, name='alpha')
    c = cp.Parameter(name='c', value=regularisation_para_c)
    N = cp.Parameter(name='n', value=n)

    # Objective to be solved. Using the definition of sum_squares()
    # Found at: https://www.cvxpy.org/tutorial/functions/index.html
    objective = cp.Maximize(cp.sum(alpha) - (0.5 * cp.sum_squares(alpha @ (label_train.reshape(-1, 1) * data_train))))

    # Constraints in the problems
    constraints = [alpha[i] >= 0 for i in range(n)] + [alpha[i] <= c / N for i in range(n)] + \
                  [alpha @ label_train.reshape(-1, 1) == 0]

    # Define and solve the problem
    prob_dual = cp.Problem(objective, constraints)
    prob_dual.solve(verbose=debug_mode)

    # Return the trained model
    return prob_dual


def primal_solutions_from_dual(data_train, label_train, data_test, label_test, svm_dual_model):
    """
    Calculate primal solutions and the accuracy of the trained SVM model using dual formulation
    :param data_train: Training features
    :param label_train: Training class label relative to training features
    :param data_test: Testing features to obtain predictions for
    :param label_test: Testing class labels relative to the testing features
    :param svm_dual_model: rained SVM model using dual formulation
    :return: Returns primal solution tuple (w*, b*)
    """

    # Fetch the trained parameter
    alpha = svm_dual_model.var_dict['alpha'].value

    # primal problem solution w*
    w = alpha @ (label_train.reshape(-1, 1) * data_train)

    # primal problem solution b*
    b_values = []
    for i, al in enumerate(alpha):
        # All the vectors satisfying the criterion alpha belongs to (0, C). Here, c=c/n.
        if (al > 0) and (al < (svm_dual_model.param_dict['c'].value/svm_dual_model.param_dict['n'].value)):
            b_values.append(label_train[i] - w @ data_train[i])

    # Averaging to get a scalar value -> sum(b_values)/len(b_values)
    b = np.mean(b_values)

    # Obtain the predictions for test data
    y_predicted = data_test @ w + b

    # Compute the predicted class
    y_predicted[y_predicted > 0] = 1
    y_predicted[y_predicted <= 0] = -1

    # Calculate the predicted accuracy
    test_accuracy = accuracy_score(label_test, y_predicted)
    print(f"[INFO] The test accuracy achieved by the SVM using dual formulation is {test_accuracy}.")

    return w, b


def support_vectors_primal(data_train, label_train, svm_model):
    """
    Returns all the support vectors based on SVM primal formulation
    :param data_train: Training features
    :param label_train: Training class label relative to training features
    :param svm_model: Trained SVM model using primal formulation
    :return: Support vectors Array
    """
    # Fetch the trained parameters
    w = svm_model.var_dict['w'].value
    b = svm_model.var_dict['b'].value

    # Fetching all the points that lie on the decision boundary based on the criteria
    # y(w.tx +b) = 1 - slack_variable
    lhs = np.multiply(label_train.reshape(-1, 1), data_train @ w + b)
    rhs = 1 - svm_model.var_dict['psi'].value

    # We won't have exactly identical but similar values
    # support_vectors = data_train[np.isclose(lhs, rhs, rtol=1e-04, atol=1e-04).flatten()]
    support_vectors = []
    for i, lhs_value in enumerate(lhs):
        # Checking for floating error
        if rhs[i] - tolerance <= lhs_value <= rhs[i] + tolerance:
            support_vectors.append(data_train[i])

    return np.stack(support_vectors)


def support_vectors_dual(data_train, svm_dual_model):
    """
    Returns all the support vectors based on SVM dual formulation
    :param data_train: Training features
    :param svm_dual_model: Trained SVM model using dual formulation
    :return: Support vectors Array
    """
    # Fetch the trained parameter
    alpha = svm_dual_model.var_dict['alpha'].value

    # Values greater than zero with some tolerance to satisfy the following conditions:
    # alpha = C or 0 < alpha < C.
    c = svm_dual_model.param_dict['c'].value / svm_dual_model.param_dict['n'].value
    support_vectors = data_train[(alpha > tolerance) & (alpha <= c + tolerance)]
    # EDIT made on 16/07/22 - alpha <= C + error. The number could be 391 if your tolerance is 1e-4.

    return support_vectors


def c_optimisation(data_train, label_train, data_cv, label_cv, data_test, label_test):
    """
    Optimisation to choose the best C value with SVM primal formulation.
    :param data_train: Training features
    :param label_train: Training class label relative to training features
    :param data_cv: Cross-validation features
    :param label_cv: Cross-validation class label relative to Cross-validation features
    :param data_test: Testing features
    :param label_test: Testing class label relative to Testing features
    :return: A tuple of (best accuracy on cv set, Optimal C value, Accuracy on test set)
    """
    # Flag variables
    val_acc = 0
    best_c = 0
    test_acc = 0

    # Range of the C values
    c_values = [2**i for i in range(-10, 11, 2)]

    for i, c_value in enumerate(c_values, start=1):
        # Train the model
        svm_model = svm_train_primal(data_train, label_train, c_value)

        # Obtain the accuracy on cv set
        acc = svm_predict_primal(data_cv, label_cv, svm_model)

        print(f"[INFO] Iteration {i}: Accuracy on cv set is {acc} for C value {c_value}.")
        if acc > val_acc:
            val_acc = acc
            best_c = c_value
            test_acc = svm_predict_primal(data_test, label_test, svm_model)

    # Return the tuple
    return val_acc, best_c, test_acc


def svm_sklearn(data_train, label_train, data_test, label_test, optimal_c):
    """
    Train and predict using Scikit-Learn SVM.
    :param data_train: Training features
    :param label_train: Training class label relative to training features
    :param data_test: Testing features
    :param label_test: Testing class label relative to Testing features
    :param optimal_c: Optimal C searched
    :return: Accuracy on the test set.
    """
    # Initialise Linear Support Vector Classification.
    # Using the primal optimisation here.
    clf = LinearSVC(random_state=42, verbose=5, C=optimal_c, max_iter=50000, dual=False)

    # Fit the model to the training data
    clf.fit(data_train, label_train)

    # Obtain the predictions
    predictions = clf.predict(data_test)

    # Calculate the accuracy
    acc = accuracy_score(label_test, predictions)

    return acc


if __name__ == '__main__':

    # Loading the data
    X_train_full, y_train_full = load_data('train.csv')
    X_test, y_test = load_data('test.csv')

    # Separate the train data into train and cross-validation set.
    # Selecting first 4000 as training set - according to question.
    # X_train, X_cv = X_train_full[:4000], X_train_full[4000:]
    X_train, X_cv = X_train_full[:4000, :], X_train_full[4000:, :]
    y_train, y_cv = y_train_full[:4000], y_train_full[4000:]

    # Clearing the memory
    del X_train_full, y_train_full

    # Global variables
    tolerance = 1e-4
    debug_mode = False

    # Ans 2 - SVM using primal formulation
    svm_model = svm_train_primal(X_train, y_train, 100)

    test_accuracy_primal = svm_predict_primal(X_test, y_test, svm_model)

    print(f"\n[INFO] The value for SVM trained with primal formulation for:\n1. b = {svm_model.var_dict['b'].value}\n"
          f"2. Sum of all w = {np.sum(svm_model.var_dict['w'].value)}")
    print(f"[INFO] The test accuracy achieved by the SVM using primal formulation is {test_accuracy_primal}.\n")

    # Ans 3 - SVM using dual formulation
    svm_dual_model = svm_train_dual(X_train, y_train, 100)
    print(f"\n[INFO] The sum of alpha for SVM trained with dual formulation is "
          f"{np.sum(svm_dual_model.var_dict['alpha'].value)}.\n")

    # Ans 4 - Primal problem solutions (w* and b*)
    primal_solutions = primal_solutions_from_dual(X_train, y_train, X_test, y_test, svm_dual_model)
    print(f"[INFO] The value for SVM trained with dual formulation for:\n1. b* = {primal_solutions[1]}\n"
          f"2. Sum of all w* = {np.sum(primal_solutions[0])}.\n")

    # Ans 5 - Support vectors from primal formulation
    sv_primal = support_vectors_primal(X_train, y_train, svm_model)
    print(f"[INFO] The support vectors for SVM primal formulation are {sv_primal}\n"
          f"Total no. of support vectors are {sv_primal.shape[0]}.\n")

    # Ans 6 - Support vectors from the dual formulation
    sv_dual = support_vectors_dual(X_train, svm_dual_model)
    print(f"[INFO] The support vectors for SVM dual formulation are {sv_dual}\n"
          f"Total no. of support vectors are {sv_dual.shape[0]}.\n")

    # Ans 7 - Optimisation
    result = c_optimisation(X_train, y_train, X_cv, y_cv, X_test, y_test)
    print(f"\n[INFO] The optimal C value is {result[1]}")
    print(f"[INFO] Test accuracy when SVM primal formulation trained with optimal C value is {result[2]}.\n")

    # Ans 8 - SVM implementation using Scikit-Learn
    # Different packages identify C differently.
    # Still using the Optimal C searched above.
    test_acc = svm_sklearn(X_train, y_train, X_test, y_test, result[2])
    print(f"\n[INFO] The test accuracy when SVM trained with the Scikit-Learn is {test_acc}.\n")

    print('END')
