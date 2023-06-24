import numpy as np
import pandas as pd
import matplotlib as mpl
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import config
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
plt.rcParams["figure.max_open_warning"] = 100

def visualize_decision_tree(model, X, y):
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
    # Plot tree
    plt.figure(figsize=(30, 20))
    tree.plot_tree(model, fontsize=10)
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_tree.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Plot feature importance
    importances = model.feature_importances_
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = np.arange(X.shape[1])
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - {}".format(model_name))
    plt.barh(range(len(indices)), importances[indices], color="r", align="center")
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=8)  # Adjust fontsize for better visibility
    plt.gca().invert_yaxis()  # Invert y-axis to show features from top to bottom
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()  # Add tight layout for better spacing
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_importance.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    else:
        print(f"Warning: {model_name} does not support probability estimates. Precision-Recall curve cannot be computed.")

def visualize_random_forest(model, X, y):
    """
    This function visualizes a random forest model.

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
    plt.title("Feature importances - {}".format(model_name))
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_feature_importances.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Plot confusion matrix
    predictions = model.predict(X)
    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix - {}".format(model_name))
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_confusion_matrix.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()


    # Convert multiclass labels to binary labels using one-vs-rest strategy
    y_binary = label_binarize(y, classes=np.unique(y))

    # Compute probabilities for each class
    prob_predictions = model.predict_proba(X)

    # Plot ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_binary.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], prob_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 6))
    for i in range(y_binary.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"ROC curve (class {i}): AUC={roc_auc[i]:0.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - {}".format(config.VISUALS_FOLDER))
    plt.legend(loc="lower right")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_roc_curve.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()


    # Plot precision-recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(y_binary.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i], prob_predictions[:, i])
        average_precision[i] = average_precision_score(y_binary[:, i], prob_predictions[:, i])
    plt.figure(figsize=(10, 6))
    for i in range(y_binary.shape[1]):
        plt.plot(recall[i], precision[i], lw=2,
                 label=f"Precision-Recall curve (class {i}): AP={average_precision[i]:0.2f}, AUC={roc_auc[i]:0.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve - {}".format(config.VISUALS_FOLDER))
    plt.legend(loc="lower left")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_precision_recall_curve.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Plot individual decision trees in the forest
    for i, tree_in_forest in enumerate(model.estimators_):
        plt.figure(figsize=(30, 20))
        tree.plot_tree(tree_in_forest, fontsize=10)
        save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_decision_tree_{i}.png')
        pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_path)
        plt.close()


def visualize_logistic_regression(model, X, y, fig_path):
    """
    This function visualizes a logistic regression model.

    Args:
        model: LogisticRegression, the logistic regression model to visualize.
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
    plt.title("Coefficients of Logistic Regression Model - {}".format(model_name))
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_coefficinets.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Plot confusion matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title("Confusion Matrix - {}".format(model_name))
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['negative', 'positive'], rotation=45)
    plt.yticks(tick_marks, ['negative', 'positive'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_confusion_matrix.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve: AUC={0:0.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - {}'.format(model_name))
    plt.savefig(fig_path)
    plt.close()

    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, model.predict_proba(X)[:, 1])
    average_precision = average_precision_score(y, model.predict_proba(X)[:, 1])
    plt.plot(recall, precision,
             label='Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - {}'.format(model_name))
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_precision_recall.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
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
    plt.title("Calibration Curve - {}".format(model_name))
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_calibration_curve.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()


def plot_classification_report(y_test, y_pred, title='Classification Report', figsize=(8, 6), dpi=70, **kwargs):
        """
        This function plots the classification report of sklearn

        Parameters
        ----------
        y_test : Series of shape (n_samples,) -> Targets.
        y_pred : Series of shape (n_samples,) -> Predictions.
        title : str, default = 'Classification Report'
        fig_size : tuple, size (inches) of the plot.
        dpi : int, Image DPI.
        **kwargs : attributes of classification_report class of sklearn

        Returns
        -------
            fig : Figure from matplotlib
            ax : Axe object from matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
        keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
        df = pd.DataFrame(clf_report, columns=keys_to_plot).T
        # the following line ensures that dataframe are sorted from the majority classes to the minority classes
        df.sort_values(by=['support'], inplace=True)

        # let's plot the heatmap by masking the 'support' column as a first step
        rows, cols = df.shape
        mask = np.zeros(df.shape)
        mask[:, cols - 1] = True

        ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", fmt='.3g',
                         vmin=0.0,
                         vmax=1.0,
                         linewidths=2, linecolor='white'
                         )

        # now let's add the support column by normalizing the colors in this column
        mask = np.zeros(df.shape)
        mask[:, :cols - 1] = True

        ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", cbar=False,
                         linewidths=2, linecolor='white', fmt='.0f',
                         vmin=df['support'].min(),
                         vmax=df['support'].sum(),
                         norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                                   vmax=df['support'].sum())
                         )

        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=360)
        save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{title}.png')
        pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)

        return fig, ax

def visualize_one_vs_rest(model, X, y):
    """
    This function visualizes the ROC curve and Precision-Recall curve for multiclass classification
    using the One-vs-Rest approach.

    Args:
        model: OneVsRestClassifier, the One-vs-Rest model to visualize.
        X: array-like, the input features.
        y: array-like, the target labels.
        fig_path: str, the path to save the generated figure.
    Returns: -
    """
    model_name = type(model).__name__

    # Convert the target labels to binary format
    y_binary = label_binarize(y, classes=model.classes_)

    # Obtain the predicted probabilities for each class
    y_score = model.predict_proba(X)

    # Calculate the false positive rate (FPR) and true positive rate (TPR) for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(model.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate the precision and recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(model.classes_)):
        precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_binary[:, i], y_score[:, i])

    # Plot the ROC curve for each class
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(model.classes_)):
        plt.plot(fpr[i], tpr[i], label='Class {}: AUC={:.2f}'.format(model.classes_[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - {}'.format(model_name))
    plt.legend(loc="lower right")

    # Plot the Precision-Recall curve for each class
    plt.subplot(1, 2, 2)
    for i in range(len(model.classes_)):
        plt.plot(recall[i], precision[i], label='Class {}: AP={:.2f}'.format(model.classes_[i], average_precision[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - {}'.format(model_name))
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_roc_pr_curves.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Compute precision and recall for each class
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)

    # Plot precision for each class
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(precision)), precision)
    plt.xticks(range(len(precision)), ['Class A', 'Class B', 'Class C', 'Class D'])
    plt.title("Precision for Each Class - {}".format(model_name))
    plt.xlabel("Class")
    plt.ylabel("Precision")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_precision.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Plot recall for each class
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(recall)), recall)
    plt.xticks(range(len(recall)), ['Class A', 'Class B', 'Class C', 'Class D'])
    plt.title("Recall for Each Class - {}".format(model_name))
    plt.xlabel("Class")
    plt.ylabel("Recall")
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_recal.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

def visualize_knn(model, X, y):
    """
    This function visualizes a K-Nearest Neighbors (KNN) model.

    Args:
        model: KNeighborsClassifier, the KNN model to visualize.
        X: array-like, the input features.
        y: array-like, the target labels.
        fig_path: str, the path to save the generated figures.
    Returns: -
    """

    model_name = type(model).__name__

    # Calculate the minimum and maximum values for each feature
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Generate a mesh grid of points using the minimum and maximum values
    h = 0.1  # Step size in the mesh
    x_values = np.arange(x_min, x_max, h)
    y_values = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(x_values, y_values)

    # Flatten the mesh grid points and repeat them for 40 features
    mesh_grid = np.column_stack((xx.ravel(), yy.ravel()))
    mesh_grid = np.repeat(mesh_grid, 40, axis=1)

    # Use the model to predict the output for the mesh grid points
    Z = model.predict(mesh_grid)

    # Reshape the predictions to match the mesh grid dimensions
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundaries - {}'.format(model_name))
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_decision_boundaries.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

    # Compute and plot precision-recall curve
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)

        # Compute one-vs-rest precision-recall curve
        precision = dict()
        recall = dict()
        for i in range(len(model.classes_)):
            if np.sum(y == i) > 0:  # Check if positive samples exist for the class
                precision[i], recall[i], _ = metrics.precision_recall_curve(y == i, y_prob[:, i])
                plt.plot(recall[i], precision[i], lw=2, label='class {}: AP={:.2f}'.format(i, average_precision[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve')
        plt.legend(loc="best")
        save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_precision_recall.png')
        pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_path)
        plt.close()
    else:
        print(f"Warning: {model_name} does not support probability estimates. Precision-Recall curve cannot be computed.")

    # Plot confusion matrix
    predictions = model.predict(X)
    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title('Confusion Matrix - {}'.format(model_name))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_fig_path = os.path.join(config.VISUALS_FOLDER, f'{model_name}_confusion_matrix.png')
    pathlib.Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fig_path)
    plt.close()

visualize_functions = {
    "decision_tree_gini": visualize_decision_tree,
    "decision_tree_entropy": visualize_decision_tree,
    "rf": visualize_random_forest,
    "logistic_regression": visualize_logistic_regression,
    "one_vs_rest": visualize_one_vs_rest,
    "knn" : visualize_knn,
}