<!-- Project Title -->
<h1 align="center">Automobile industry customer segmentation and targeted marketing</h1>

<!-- Table of Contents -->
## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Results](#results)
- [Project Organization](#project-organization)
- [Usage and Instructions](#usage-and-instructions)
- [License](#license)
- [Contributing](#contributing)

<!-- Project Overview -->
# Project Overview:

This project aims to assist an automobile company in predicting the appropriate customer segment for new potential customers in a new market. The company has already established four customer segments (A, B, C, D) in their existing market and has successfully implemented targeted marketing strategies for each segment.
They now plan to expand into a new market and want to leverage their existing customer segmentation strategy.

The company has conducted extensive market research and identified 2,627 potential new customers in the new market. The objective is to build a predictive model that can accurately classify these new customers into the appropriate customer segment (A, B, C, or D) based on their characteristics and behavior.

<!--Problem Statement-->
# Problem Statement:

The goal of this project is to develop a machine learning model that can predict the customer segment (A, B, C, or D) for new potential customers in the new market. The model will be trained on the existing customer data, including the customer segments and their corresponding characteristics. The model will then be used to predict the customer segment for the new potential customers based on their available information.

<!--Results: -->
# Results:

Please refer to the exploration notebook for an extensive variable analysis.

Based on the precision, recall, and accuracy metrics obtained from the model's performance in classifying potential new customers into the existing customer segments (A, B, C, D), the most performant model is Random Forest classifier.

Precision: Precision measures the accuracy of the positive predictions made by the model. It represents the proportion of correctly classified potential new customers in each customer segment out of all the customers predicted to belong to that segment.

For segment A, the precision is approximately 0.458, indicating that around 45.8% of the customers predicted to be in segment A by the model are actually in segment A.

For segment B, the precision is approximately 0.476, suggesting that roughly 47.6% of the customers predicted to be in segment B are actually in segment B.

For segment C, the precision is approximately 0.471, implying that approximately 47.1% of the customers predicted to be in segment C are actually in segment C.

For segment D, the precision is approximately 0.487, indicating that about 48.7% of the customers predicted to be in segment D are actually in segment D.

Higher precision values for each segment suggest a lower rate of false positives, meaning that the model is correctly identifying a significant proportion of potential customers who truly belong to each segment.

Recall: Recall, also known as sensitivity or true positive rate, measures the model's ability to correctly identify all positive instances in each customer segment. It represents the proportion of actual customers in each segment that the model correctly identifies.

For segment A, the recall is approximately 0.465, suggesting that the model correctly captures around 46.5% of the actual customers in segment A.

For segment B, the recall is approximately 0.478, indicating that the model correctly captures approximately 47.8% of the actual customers in segment B.

For segment C, the recall is approximately 0.470, indicating that the model correctly captures around 47.0% of the actual customers in segment C.

For segment D, the recall is approximately 0.491, indicating that the model correctly captures approximately 49.1% of the actual customers in segment D.

Higher recall values for each segment suggest a lower rate of false negatives, meaning that the model is effectively capturing a significant proportion of actual customers belonging to each segment.

Accuracy: Accuracy is a measure of the overall correctness of the model's predictions across all customer segments. It represents the proportion of correctly classified instances (both true positives and true negatives) out of all the instances.

The accuracy of the model in classifying potential new customers into the appropriate customer segments is approximately 0.465. This indicates that the model correctly classifies approximately 46.5% of all potential new customers in the new market. This is not a dependable metric however.

<!--Project Organization: -->
# Project Organization: 
### The project follows the following directory structure:

    ├── input
    │   ├── Train.csv
    │   ├── dataset.csv
    │   ├── encoded_data.csv
    │   ├── ....
    ├── notebooks
    │   ├── exploration.ipynb
    ├── models
    │   ├── decision_tree_entropy_0.bin
    │   ├── ....
    ├── src
    │   ├── config.py
    │   ├── data_cleaning.py
    │   ├── data_encoding.py
    │   ├── data_scaler.py
    │   ├── main.py
    │   └── ....
    ├── tests
    │   ├── test_data_encoding.py
    │   ├── ....
    ├── visualizations
    │   ├── data_before_and_after_cleaning.png
    │   ├── ....
    ├── README.md

* data: Directory to store the input data files but also data in the different stages of the project (e.g., encoded data).
* notebooks: All Jupyter notebooks (i.e. any *.ipynb file) are stored in the notebooks folder.
* models: Directory to save the trained machine learning models.
* src: This folder keeps all the python scripts associated with the project here. If a python script is mentioned, i.e. any *.py file, it is probably stored in the src folder.
* tests: Contains unit tests for the source code to ensure its correctness and functionality.
* visualizations: Directory to save all graphs related to the project.
* README.md: Documentation file providing an overview of the project, its purpose, and instructions for usage.

<!-- Usage and Instructions -->
# Usage and Instructions
### 1. Clone the repository:
 
```
git clone <repository-url>
```

### 2. Navigate to "src"
```
git cd src
```
### 3. Run the program 

```
python main.py
```
The number of created folds is by default 5.
If you want to specify otherwise you may do so by (example for 2 splits):
```
python main.py --n_splits 2
```

You also may want to choose a specific model to use. In this case, you can do it by typing:

```
python main.py --n_splits 5 --model decision_tree_entropy
```

The available models are :
    "logistic_regression",
    "decision_tree_gini",
    "decision_tree_entropy",
    "rf", (for Random Forest)
    "naive_bayes", 
    "one_vs_rest", (Logistic Regression as a default model)
    "one_vs_one", (Linear Support Vector Classification)
    "knn".

### 4. Review the results:

Analyze the predicted customer segments for the new potential customers and use them for targeted marketing strategies.

You may navigate into the /models section to check for the train models as well as its hyperameters and investigate the best ones.
In addition to this, the /visualizations section provides the corresponding visuals for the trained model. 

### 5. Modify and extend:

Feel free to modify the code and adapt it to your specific needs.
Add additional tests in the tests directory to ensure the correctness of the code.
Update the documentation and README.md file to reflect any changes made.

<!--License -->
# License:

This project is licensed under the MIT License. Feel free to use and modify the code according to the terms of the license.

<!--Contributing -->
# Contributing: 

Contributions to this project are welcome. Feel free to open issues and submit pull requests to suggest improvements, add new features, or fix any bugs.
