import numpy as np
import pandas as pd
import matplotlib as mpl
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report
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

def plot_classification_report(y_test, y_pred, title='Classification Report', figsize=(8, 6), dpi=70,
                                   save_fig_path=None, **kwargs):
        """
        This function plots the classification report of sklearn

        Parameters
        ----------
        y_test : pandas.Series of shape (n_samples,)
            Targets.
        y_pred : pandas.Series of shape (n_samples,)
            Predictions.
        title : str, default = 'Classification Report'
            Plot title.
        fig_size : tuple, default = (8, 6)
            Size (inches) of the plot.
        dpi : int, default = 70
            Image DPI.
        save_fig_path : str, defaut=None
            Full path where to save the plot. Will generate the folders if they don't exist already.
        **kwargs : attributes of classification_report class of sklearn

        Returns
        -------
            fig : Matplotlib.pyplot.Figure
                Figure from matplotlib
            ax : Matplotlib.pyplot.Axe
                Axe object from matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
        keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
        df = pd.DataFrame(clf_report, columns=keys_to_plot).T
        # the following line ensures that dataframe are sorted from the majority classes to the minority classes
        df.sort_values(by=['support'], inplace=True)

        # first, let's plot the heatmap by masking the 'support' column
        rows, cols = df.shape
        mask = np.zeros(df.shape)
        mask[:, cols - 1] = True

        ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", fmt='.3g',
                         vmin=0.0,
                         vmax=1.0,
                         linewidths=2, linecolor='white'
                         )

        # then, let's add the support column by normalizing the colors in this column
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

        if (save_fig_path != None):
            path = pathlib.Path(save_fig_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_fig_path)

        return fig, ax

visualize_functions = {
    "decision_tree_gini": visualize_decision_tree,
    "decision_tree_entropy": visualize_decision_tree,
    "rf": visualize_random_forest,
    "logistic_regression": visualize_logistic_regression,
}