<!-- Project Title -->
<h1 align="center">Customer Segmentation and Targeted Marketing</h1>

<!-- Table of Contents -->
## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
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

<!--Project Organization: -->
# Project Organization: 
### The project follows the following directory structure:

    ├── data
    │   ├── Train.csv
    │   ├── data_cleaned.csv
    │   ├── encoded_data.csv
    │   ├── ....
    ├── notebooks
    │   ├── exploration.ipynb
    ├── models
    │   ├── ...
    │   ├── ...
    │   └── ...
    ├── src
    │   ├── config.py
    │   ├── data_cleaning.py
    │   ├── data_encoding.py
    │   ├── data_scaler.py
    │   ├── main.py
    │   └── ...
    ├── tests
    │   ├── test_data_encoding.py
    │   ├── ...
    │   ├── ...
    │   ├── ...
    │   └── ...
    ├── visualizations
    │   ├── data_before_and_after_cleaning.png
    │   ├── ...
    │   └── ...
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
### 3. Run main.py 

```
python src/main.py
```

### 4. Review the results:

Analyze the predicted customer segments for the new potential customers and use them for targeted marketing strategies.

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
