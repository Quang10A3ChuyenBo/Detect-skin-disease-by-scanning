import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVM:
    def __init__(self, kernel='linear', C=1.0, random_state=42):
        self.svm = SVC(kernel=kernel, C=C, random_state=random_state)
        self.best_params_ = None

    def train(self, X_train, y_train):
        self.svm.fit(X_train, y_train)

    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5):
        grid_search = GridSearchCV(SVC(random_state=self.svm.random_state), param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        self.svm = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

    def predict(self, X):
        return self.svm.predict(X)