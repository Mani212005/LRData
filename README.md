# LRData

# Unified Data Analysis and ML Modeling Platform (InsightiGraph) 📊

Welcome to **InsightiGraph**, your all-in-one web application for effortless data exploration, visualization, and machine learning model training. This platform simplifies the process of gaining insights from your data and building robust predictive models, all through an intuitive and interactive interface.

Whether you're a data analyst, a student, or a seasoned data scientist, InsightiGraph empowers you to:

* **Quickly understand your datasets** with automated data overviews and descriptive statistics.
* **Generate stunning, interactive visualizations** with just a few clicks.
* **Perform comprehensive Automated Exploratory Data Analysis (EDA)** reports.
* **Build and evaluate Linear Regression models** with customizable preprocessing steps.

![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr%201.png)

---

## ✨ Features

### Data Explorer

* **Interactive Data Upload:** Easily upload your CSV files.
* **Sample Datasets:** Get started instantly with pre-loaded popular datasets like Boston Housing, California Housing, Medical Insurance Costs, Fish Market, and Salary Data.
* **Data Overview:**
    * Key metrics (rows, columns, numeric/text columns).
    * Data preview and type inference.
    * Descriptive statistics.
    * Missing value analysis.
* **Visualization Studio:**
    * A wide array of interactive plots powered by Plotly and Seaborn:
        * Scatter Plots
        * Line Plots
        * Bar Charts
        * Histograms
        * Box Plots
        * Pie Charts
        * Heatmaps (Correlation Matrix)
        * Pair Plots
        * Area Charts
        * Violin Plots
        * Strip Plots
    * Customizable chart options (X/Y axis, color, bins for histograms).
    * Download plots as PNG or HTML.
* **Automated EDA Report:** Generate a comprehensive and interactive `ydata-profiling` report with a single click, providing deep insights into your data's characteristics, relationships, and potential issues.
* **Processed Data Download:** Download your current dataset (after any re-naming if using sample data) as a CSV.

### ML Modeler

* **User-Friendly Interface:** Guide you through the model training process step-by-step.
* **Preprocessing Options:**
    * **Missing Value Handling:** Choose between dropping rows, mean imputation, or median imputation.
    * **Outlier Handling:** Option to remove outliers using the Interquartile Range (IQR) method.
    * **Categorical Feature Encoding:** Automatic One-Hot Encoding for selected categorical columns.
* **Feature & Target Selection:** Intuitive selection of independent (features) and dependent (target) variables.
* **Model Selection:** Choose between standard Linear Regression, Ridge Regression, or Lasso Regression.
* **Hyperparameter Tuning:** Adjust the `alpha` parameter for Ridge and Lasso models.
* **Model Performance Metrics:** View key metrics such as:
    * Coefficients and Intercept
    * $R^2$ Score
    * Mean Absolute Error (MAE)
    * Mean Squared Error (MSE)
    * Cross-Validation $R^2$ (Mean and Standard Deviation)
* **Model Visualizations:**
    * **Predicted vs. Actual Values Plot:** Understand how well your model's predictions align with the actual values.
    * **Residual Plot:** Analyze the errors of your model to detect patterns or biases.
    * **Feature Importance (Coefficients) Plot:** Visualize the impact of each feature on the target variable.
* **Download Options:** Download the trained model (`.pkl`) and the predictions (`.csv`).

---

## 🚀 Getting Started

To run this application locally, follow these steps:

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download the script):**

    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

    *(Replace `<repository_url>` and `<repository_folder>` with your actual repository details if hosted on GitHub/GitLab/etc. If you only have the `app.py` file, simply navigate to its directory.)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install streamlit pandas plotly seaborn matplotlib scikit-learn ydata-profiling
    ```

### Running the Application

1.  **Navigate to the directory** containing your Streamlit application file (e.g., `app.py`).

2.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser.
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr2.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr3.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr4.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr5.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr6.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr7.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr8.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr9.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr10.png)
    ![Image_Alt](https://github.com/Mani212005/LRData/blob/6d4bd286991fddb9d9129966f0029d33b8b147da/lr11.png)

