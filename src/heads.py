"""The 4 classification heads and their hyperparameter grids (same grid for every representation).

- Logistic regression: linear, measures how separable a representation is
- SVM with RBF kernel: standard strong baseline in e-nose work
- Random forest: nonlinear, gives feature importance
- MLP with 1 hidden layer: small neural head

Hyperparameters are chosen on an inner validation split of the training fold (see cv.py),
never on the test fold. class_weight="balanced" (where available) handles the unbalanced
binary tasks (60 / 540 and 54 / 101); it is the same for every representation.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

HEAD_NAMES = ["logreg", "svm", "rf", "mlp"]


def make_heads(seed):
    """Fresh, unfitted heads with their grids.

    returns {name: (estimator, grid)}; grid keys are estimator parameter names
    """
    return {
        "logreg": (LogisticRegression(max_iter=5000, class_weight="balanced"),
                   {"C": [0.01, 0.1, 1, 10]}),
        "svm": (SVC(kernel="rbf", class_weight="balanced"),
                {"C": [1, 10, 100], "gamma": ["scale", 0.01]}),
        "rf": (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed),
               {"max_depth": [None, 10]}),
        "mlp": (MLPClassifier(max_iter=1000, random_state=seed),
                {"hidden_layer_sizes": [(64,), (128,)], "alpha": [1e-4, 1e-2]}),
    }
