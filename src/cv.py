"""Cross validation per dataset, metrics, and writing results/.

Will provide:
- folds: GasWindTunnel StratifiedKFold(5) stratified by gas x wind speed (480 train / 120 test);
  EverydayScents StratifiedGroupKFold(5), groups = session (about 124 train / 31 test); 3 seeds each;
- the fold loop: StandardScaler fitted on F[train_idx] only, then head fitted on the scaled training fold;
- metrics: macro F1 (primary), balanced accuracy, ROC AUC for binary tasks, confusion matrix;
- writing one csv per notebook to results/ with columns: seed, fold, task, rep, head, metrics.

F: (N, D) features, y: (N,) labels.
"""
