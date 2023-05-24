import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
plt.rcParams["figure.max_open_warning"] = 100

def visualize_decision_tree(model, X, y, fig_path):
    """
    This function visualizes a decision tree model.

    Args:
        model: DecisionTreeClassifier, the decision tree model to visualize.
        X: array-like, the input features.
        y: array-like, the target labels.
        fig_path: str, the path to save the generated figures.
    Returns: -
    """

    model_name = type(model).__name__
    #Plot tree
    plt.figure(figsize=(30,20))
    tree.plot_tree(model, fontsize=10)
    plt.savefig(fig_path + '_tree.png')
    plt.close()

    # Plot feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances - {}".format(fig_path))
    plt.bar(range(len(indices)), importances[indices], color="r", align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.savefig(f'{model_name}_importance.png')
    plt.close()

def visualize_random_forest(model, X, y, fig_path):
    """
    This functioon isualizes a random forest model.

    Args:
        model: RandomForestClassifier, the random forest model to visualize.
        X: array-like, the input features.
        y: array-like, the target labels.
        fig_path: str, the path to save the generated figures.
    Returns: -
    """
    model_name = type(model).__name__

    # Get feature importances from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importances'):
        importances = model.get_feature_importances()
    else:
        raise AttributeError("The model does not have attribute 'feature_importances_' or 'get_feature_importances'.")

    # Plot feature importances
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature importances - {}".format(fig_path))
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig(f'{model_name}_feature_importances.png')
    plt.close()

    # Plot confusion matrix
    predictions = model.predict(X)
    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix - {}".format(fig_path))
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('Receiver Operating Characteristic - {}'.format(fig_path))
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()

    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y, predictions)
    average_precision = average_precision_score(y, predictions)
    plt.figure(figsize=(10, 6))
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("2-class Precision-Recall curve: AP={0:0.2f}".format(average_precision).format(fig_path))
    plt.savefig(f'{model_name}_precision_recall_curve.png')
    plt.close()

    # Plot individual decision trees in the forest
    for i, tree_in_forest in enumerate(model.estimators_):
        plt.figure(figsize=(30,20))
        tree.plot_tree(tree_in_forest, fontsize=10)

def visualize_logistic_regression(model, X, y, fig_path):
    """
    This function visualizes a logistic regression model.

    Args:
        model: RandomForestClassifier, the random forest model to visualize.
        X: array-like, the input features.
        y: array-like, the target labels.
        fig_path: str, the path to save the generated figures.
    Returns: -
    """

    model_name = type(model).__name__
    # Plot coefficients
    coef = model.coef_.ravel()
    plt.figure(figsize=(15, 5))
    plt.stem(coef)
    plt.xticks(range(len(coef)), range(len(coef)))
    plt.title("Coefficients of Logistic Regression Model - {}".format(fig_path))
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.savefig(f'{model_name}_coefficients.png')
    plt.close()

    # Plot confusion matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title("Confusion Matrix - {}".format(fig_path))
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['negative', 'positive'], rotation=45)
    plt.yticks(tick_marks, ['negative', 'positive'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:,1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - {}'.format(fig_path))
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()

    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, model.predict_proba(X)[:,1])
    plt.plot(recall, precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - {}'.format(fig_path))
    plt.savefig(f'{model_name}_precision_recall_curve.png')
    plt.close()

    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_pos = model.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model.__class__.__name__}")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title("Calibration Curve - {}".format(fig_path))
    plt.savefig(f'{model_name}_calibration_curve.png')
    plt.close()

visualize_functions = {
    "decision_tree_gini": visualize_decision_tree,
    "decision_tree_entropy": visualize_decision_tree,
    "rf": visualize_random_forest,
    "logistic_regression": visualize_logistic_regression
}