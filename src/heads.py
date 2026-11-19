"""The 4 classification heads and their hyperparameter grids (same grid for every representation).

- Logistic regression: linear, measures how separable a representation is
- SVM with RBF kernel: standard strong baseline in e-nose work
- Random forest: nonlinear, gives feature importance
- MLP with 1 hidden layer: small neural head

Hyperparameters are chosen on an inner validation split of the training data, never on the test fold.
"""
