from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
models = {
    "decision_tree_gini" : tree.DecisionTreeClassifier(
        criterion = "gini"
    ),
    "decision_tree_entropy" : tree.DecisionTreeClassifier(
        criterion = "entropy"
    ),
    "rf" : ensemble.RandomForestClassifier(n_estimators = 100, n_jobs=-1),
    "logistic_regression" : linear_model.LogisticRegression()
}