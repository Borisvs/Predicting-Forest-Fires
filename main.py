import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from forestFireDataset import ForestFireDataset


def main():
    dataset = ForestFireDataset("forestfires.csv")

    # Exploratory Data Analysis
    print("\n--- Exploratory Data Analysis ---\n")
    dataset.eda()
    dataset.process_data()

    # Kernel creation and setup with optimized hyperparameters.
    lbound = 1e-2
    rbound = 1e15
    _, n_features = dataset.x_train.shape
    kernel = C(0.3, (lbound, rbound)) * RBF(n_features * [10], (lbound, rbound))

    models = []

    # Gaussian Regression in order to predict the burned area of the fire.
    models.append(("Gaussian Regression", GaussianProcessRegressor(kernel=kernel,
                                                                   random_state=1,
                                                                   n_restarts_optimizer=10,
                                                                   alpha=0.001)))
    # Ridge Regression with built-in cross-validation

    models.append(("Ridge Regression", RidgeCV(cv=5, fit_intercept=False)))

    # A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of
    # the dataset and uses averaging to improve the predictive accuracy.
    models.append(("Random Forest Regressor", RandomForestRegressor()))

    # Epsilon-Support Vector Regression.
    models.append(("Support Vector Machine Regressor", SVR()))

    # Binary Classifications in order to predict if there will be a fire based on the provided circumstances.
    models.append(("Random Forest Classification", RandomForestClassifier()))
    models.append(("Support Vector Machine Classification", SVC(C=1.25, kernel="sigmoid")))
    stats = pd.DataFrame(columns=['Model', 'R2', 'MAE'])
    r2 = []
    for name, model in models:
        if "Classification" in name:
            dataset.cast_targets()

        model.fit(dataset.x_train, dataset.y_train)

        print(f"\n--- {name} ---\n")
        print(f"R-squared score: {model.score(dataset.x_train, dataset.y_train)}")
        scores = [(value - dataset.y_test[index]) for index, value in enumerate(model.predict(dataset.x_test))]
        print(f"Mean Absolute Error: {np.abs(np.sum(scores) / len(scores))}")



if __name__ == "__main__":
    main()
