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

* Summarized CV evaluation metrics on the training set (per model): 

| Model                               | Fold  | Precision   | Recall      | Accuracy    |
|-------------------------------------|-------|-------------|-------------|-------------|
| Decision Tree (Entropy)             |   0   | 0.428       | 0.413       | 0.413       |
| Decision Tree (Entropy)             |   1   | 0.441       | 0.438       | 0.438       |
| Decision Tree (Entropy)             |   2   | 0.422       | 0.420       | 0.420       |
| Decision Tree (Entropy)             |   3   | 0.476       | 0.466       | 0.466       |
| Decision Tree (Entropy)             |   4   | 0.442       | 0.437       | 0.437       |
| -------------------------------     |-------|-------------|-------------|-------------|
| Decision Tree (Gini)                |   0   | 0.442       | 0.444       | 0.444       |
| Decision Tree (Gini)                |   1   | 0.446       | 0.442       | 0.442       |
| Decision Tree (Gini)                |   2   | 0.429       | 0.425       | 0.425       |
| Decision Tree (Gini)                |   3   | 0.429       | 0.426       | 0.426       |
| Decision Tree (Gini)                |   4   | 0.420       | 0.416       | 0.416       |
| -------------------------------     |-------|-------------|-------------|-------------|
| k-Nearest Neighbors (KNN)           |   0   | 0.470       | 0.468       | 0.468       |
| k-Nearest Neighbors (KNN)           |   1   | 0.476       | 0.463       | 0.463       |
| k-Nearest Neighbors (KNN)           |   2   | 0.462       | 0.456       | 0.456       |
| k-Nearest Neighbors (KNN)           |   3   | 0.457       | 0.453       | 0.453       |
| k-Nearest Neighbors (KNN)           |   4   | 0.475       | 0.472       | 0.472       |
| -------------------------------     |-------|-------------|-------------|-------------|
| Naive Bayes                         |   0   | 0.488       | 0.492       | 0.492       |
| Naive Bayes                         |   1   | 0.480       | 0.486       | 0.486       |
| Naive Bayes                         |   2   | 0.484       | 0.488       | 0.488       |
| Naive Bayes                         |   3   | 0.473       | 0.483       | 0.483       |
| Naive Bayes                         |   4   | 0.480       | 0.497       | 0.497       |
| -------------------------------     |-------|-------------|-------------|-------------|
| One-vs-One (Linear SVC)             |   0   | 0.497       | 0.503       | 0.503       |
| One-vs-One (Linear SVC)             |   1   | 0.516       | 0.523       | 0.523       |
| One-vs-One (Linear SVC)             |   2   | 0.496       | 0.507       | 0.507       |
| One-vs-One (Linear SVC)             |   3   | 0.507       | 0.511       | 0.511       |
| One-vs-One (Linear SVC)             |   4   | 0.469       | 0.478       | 0.478       |
| -------------------------------     |-------|-------------|-------------|-------------|
| One-vs-Rest (Logistic Regression)   | 0   | 0.479       | 0.494       | 0.494       |
| One-vs-Rest (Logistic Regression)   | 1   | 0.517       | 0.525       | 0.525       |
| One-vs-Rest (Logistic Regression)   | 2   | 0.492       | 0.499       | 0.499       |
| One-vs-Rest (Logistic Regression)   | 3   | 0.476       | 0.485       | 0.485       |
| One-vs-Rest (Logistic Regression)   | 4   | 0.485       | 0.498       | 0.498       |
| -------------------------------     |-------|-------------|-------------|-------------|
| Random Forest                       |   0   | 0.458       | 0.465       | 0.465       |
| Random Forest                       |   1   | 0.476       | 0.478       | 0.478       |
| Random Forest                       |   2   | 0.471       | 0.470       | 0.470       |
| Random Forest                       |   3   | 0.487       | 0.491       | 0.491       |
| Random Forest                       |   4   | 0.482       | 0.483       | 0.483       |


* Summarized test data classification report using different hyperparameters on each model: 

| Model                             | Class | Precision   | Recall      | F1-Score    | Support     |
|-----------------------------------|-------|-------------|-------------|-------------|-------------|
| Decision Tree (Gini)              |   A   | 0.457       | 0.391       | 0.422       | 450.0       |
| Decision Tree (Gini)              |   B   | 0.431       | 0.343       | 0.382       | 463.0       |
| Decision Tree (Gini)              |   C   | 0.547       | 0.692       | 0.611       | 509.0       |
| Decision Tree (Gini)              |   D   | 0.686       | 0.716       | 0.701       | 557.0       |
| -------------------------------   |-------|-------------|-------------|-------------|-------------|
| Decision Tree (Entropy)           |   A   | 0.450       | 0.447       | 0.448       | 450.0       |
| Decision Tree (Entropy)           |   B   | 0.430       | 0.244       | 0.311       | 463.0       |
| Decision Tree (Entropy)           |   C   | 0.527       | 0.680       | 0.593       | 509.0       |
| Decision Tree (Entropy)           |   D   | 0.667       | 0.732       | 0.698       | 557.0       |
| -------------------------------   |-------|-------------|-------------|-------------|-------------|
| Naive Bayes                       |   A   | 0.423       | 0.467       | 0.444       | 450.0       |
| Naive Bayes                       |   B   | 0.430       | 0.179       | 0.253       | 463.0       |
| Naive Bayes                       |   C   | 0.483       | 0.707       | 0.574       | 509.0       |
| Naive Bayes                       |   D   | 0.659       | 0.645       | 0.652       | 557.0       |
| -------------------------------   |-------|-------------|-------------|-------------|-------------|
| k-Nearest Neighbors (KNN)         |   A   | 0.791       | 0.900       | 0.842       | 450.0       |
| k-Nearest Neighbors (KNN)         |   B   | 0.819       | 0.812       | 0.816       | 463.0       |
| k-Nearest Neighbors (KNN)         |   C   | 0.848       | 0.835       | 0.842       | 509.0       |
| k-Nearest Neighbors (KNN)         |   D   | 0.963       | 0.876       | 0.917       | 557.0       |
| -------------------------------   |-------|-------------|-------------|-------------|-------------|
| One-vs-One (Linear SVC)           |   A   | 0.438       | 0.476       | 0.456       | 450.0       |
| One-vs-One (Linear SVC)           |   B   | 0.454       | 0.289       | 0.354       | 463.0       |
| One-vs-One (Linear SVC)           |   C   | 0.555       | 0.664       | 0.605       | 509.0       |
| One-vs-One (Linear SVC)           |   D   | 0.672       | 0.707       | 0.689       | 557.0       |
| -------------------------------   |-------|-------------|-------------|-------------|-------------|
| One-vs-Rest (Logistic Regression) | A | 0.450       | 0.416       | 0.432       | 450.0       |
| One-vs-Rest (Logistic Regression) | B | 0.451       | 0.259       | 0.329       | 463.0       |
| One-vs-Rest (Logistic Regression) | C | 0.533       | 0.707       | 0.608       | 509.0       |
| One-vs-Rest (Logistic Regression) | D | 0.654       | 0.731       | 0.690       | 557.0       |
| -------------------------------   |-------|-------------|-------------|-------------|-------------|
  |


Based on the precision, recall, and accuracy metrics obtained from the model's performance in classifying potential new customers into the existing customer segments (A, B, C, D), the most performant model is Random Forest classifier.

Given that precision measure the proportion of correctly classified instances of a particular customer segment out of all instances as that class, the higher the precision the fewer the false positives.
In this project, precision translates to the accuracy of the model in correctly potential customers belong to a specific segment ouf of A, B, C, D. 

Recall measures the proportion of correctly predicted instances of a particular customer segment out of all instances belonging to that class.
In this project, recall represents the model's ability to identify and capture all potential customers from a specific segment.
In comparison to precision this evaluation metric is slightly different. In simple terms, higher recall indicates that the model is likely to capture a larger proportion of the potential customers withing a segment. 

The F1 score is calculated as the harmonic mean of precision and recall, giving equal importance to both metrics. In the context of predicting customer segments for potential new customers, the F1 score provides an overall assessment of the model's effectiveness in correctly classifying instances while considering both false positives and false negatives. 
A higher F1 score indicates better performance in achieving a balance between precision and recall. 

Lastly, although this a lot of times is not a dependable metric, accuracy measures the overall correctness of the model predictions across all segments.
In this project, accuracy the model's ability to correctly predict the appropriate customer segment for potential customers in the new market.

Overall, the model exhibits moderate scores on both the training and test set. An exceptional model is crucial for the automobile company as it enables targeted marketing strategies tailored to each customer segment, leading to more effective and personalized marketing campaigns without errors.
The high recall scores ensures that the company does not miss out on potential opportunities and can maximize its outreach to the target audience in the new market.

Based on the feature importance of the Random Forest model on the training set we can derive the following:

* Profession_Marketing: This feature has the highest importance score of 0.08. It suggests that the profession of the potential customers, specifically those in the marketing field, plays a significant role in predicting their customer segment. This feature indicates that customers working in marketing-related professions have distinct characteristics and behaviors that differentiate them across segments.

* Gender: The gender feature has an importance score of 0.073, indicating its relevance in predicting customer segments. Gender can be a meaningful differentiating factor in customer behavior and preferences, influencing the segmentation process. The importance of gender suggests that there are notable variations in customer segments based on this characteristic.

* Graduated: The graduated feature has an importance score of 0.05, suggesting that the educational background of potential customers contributes to their classification into specific segments. This feature highlights the impact of customers' educational attainment on their segment classification.

By understanding the importance of these features, the automobile company can develop targeted marketing strategies based on the unique characteristics and behaviors associated with each segment.

However, it's important to note that these three features represent only a subset of the features used in the Random Forest model!

In regard to the AUC curve of Random Forest, the AUC score is 0.85 that indicates that the model has a high discriminatory power for segment D. On the contrary, for segment B, AUC score is only 0.62 and assumes that model may face some challenges when trying to classify customers belonging to the B segment.
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
