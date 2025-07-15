import streamlit as st
import pandas as pd
import os
from pathlib import Path
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit.components.v1 as components
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Unified Data & ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a modern UI ---
st.markdown("""
<style>
    /* General Styles */
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a; /* Dark Blue */
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .section-header {
        font-size: 1.75rem;
        font-weight: bold;
        color: #3b82f6; /* Bright Blue */
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-card .stMetricLabel {
        font-size: 1.1rem;
        font-weight: bold;
        color: #4b5563; /* Gray */
    }
    .metric-card .stMetricValue {
        font-size: 2.5rem;
        color: #1e3a8a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        border-bottom: 2px solid #e0e0e0;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #3b82f6;
        color: #3b82f6;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
    }
    .welcome-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 4rem;
        border-radius: 1rem;
        text-align: center;
    }
    .welcome-container h1 {
        font-size: 3.5rem;
        font-weight: bold;
    }
    .welcome-container p {
        font-size: 1.25rem;
        max-width: 700px;
        margin: 1rem auto;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Main Application Title ---
st.title("Unified Data Analysis and ML Modeling Platform")

# --- Helper Functions (DataGraph) ---

def display_welcome_message():
    """Shows a visually appealing welcome screen."""
    st.markdown("""
    <div class="welcome-container">
        <h1>✨ Welcome to InsightiGraph!</h1>
        <p>Your intelligent assistant for automated Exploratory Data Analysis (EDA) and stunning visualizations. 
        Simply upload your CSV file to unlock insights and create beautiful, interactive charts in seconds.</p>
        <p><strong>Get started by dragging and dropping your file on the left.</strong></p>
    </div>
    """, unsafe_allow_html=True)

def display_data_overview(df):
    """Displays data metrics, preview, and statistics."""
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card">📈<br><b>Rows</b><br><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card">📊<br><b>Columns</b><br><h2>{len(df.columns)}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card">🔢<br><b>Numeric Cols</b><br><h2>{len(df.select_dtypes(include=["number"]).columns)}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card">📝<br><b>Text Cols</b><br><h2>{len(df.select_dtypes(include=["object"]).columns)}</h2></div>', unsafe_allow_html=True)

    # Data Preview & Info
    with st.expander("📋 Data Preview and Types", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"**Shape of the dataset:** {df.shape}")

    # Statistics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.markdown("### 🔍 Missing Values")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Count']
        missing_data['Missing %'] = (missing_data['Missing Count'] / len(df)) * 100
        st.dataframe(missing_data[missing_data['Missing Count'] > 0], use_container_width=True)
        if missing_data['Missing Count'].sum() == 0:
            st.success("✅ No missing values found!")

def display_visualizations(df):
    """Handles the creation and display of various plots."""
    st.markdown('<h2 class="section-header">🎨 Visualization Studio</h2>', unsafe_allow_html=True)
    
    graph_type = st.selectbox(
        'Select Graph Type',
        ['Scatter Plot', 'Line Plot', 'Bar Chart', 'Histogram', 'Box Plot', 'Pie Chart', 'Heatmap', 'Pair Plot', 'Area Chart', 'Violin Plot', 'Strip Plot']
    )

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()

    fig = None
    # --- Plotting Logic ---
    if graph_type == 'Scatter Plot':
        st.subheader("Scatter Plot Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox('X-Axis', all_cols, key='scatter_x')
        with col2:
            y_col = st.selectbox('Y-Axis', all_cols, key='scatter_y')
        with col3:
            color_col = st.selectbox('Color By (Optional)', ['None'] + all_cols, key='scatter_color')
        
        color_arg = None if color_col == 'None' else color_col
        fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, title=f'{x_col} vs. {y_col}')

    elif graph_type == 'Line Plot':
        st.subheader("Line Plot Options")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox('X-Axis', all_cols, key='line_x')
        with col2:
            y_col = st.selectbox('Y-Axis', all_cols, key='line_y')
        fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} over {x_col}')

    elif graph_type == 'Bar Chart':
        st.subheader("Bar Chart Options")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox('X-Axis', all_cols, key='bar_x')
        with col2:
            y_col = st.selectbox('Y-Axis', all_cols, key='bar_y')
        fig = px.bar(df, x=x_col, y=y_col, title=f'Average {y_col} by {x_col}')

    elif graph_type == 'Histogram':
        st.subheader("Histogram Options")
        col1, col2 = st.columns(2)
        with col1:
            col = st.selectbox('Select Column', numeric_cols, key='hist_col')
        with col2:
            bins = st.slider('Number of Bins', 5, 100, 20, key='hist_bins')
        fig = px.histogram(df, x=col, nbins=bins, title=f'Distribution of {col}')

    elif graph_type == 'Box Plot':
        st.subheader("Box Plot Options")
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox('Y-Axis', numeric_cols, key='box_y')
        with col2:
            x_col = st.selectbox('X-Axis (Optional)', ['None'] + categorical_cols, key='box_x')
        color_arg = None if x_col == 'None' else x_col
        fig = px.box(df, y=y_col, x=color_arg, title=f'Box Plot of {y_col}')

    elif graph_type == 'Pie Chart':
        st.subheader("Pie Chart Options")
        col = st.selectbox('Select Column', categorical_cols, key='pie_col')
        counts = df[col].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title=f'Distribution of {col}')

    elif graph_type == 'Heatmap':
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        else:
            st.warning("Heatmap requires at least 2 numeric columns.")

    elif graph_type == 'Pair Plot':
        if len(numeric_cols) >= 2:
            st.info("Generating Pair Plot... This might take a moment.")
            pair_plot_fig = sns.pairplot(df[numeric_cols])
            st.pyplot(pair_plot_fig)
        else:
            st.warning("Pair Plot requires at least 2 numeric columns.")

    elif graph_type == 'Area Chart':
        st.subheader("Area Chart Options")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox('X-Axis', all_cols, key='area_x')
        with col2:
            y_col = st.selectbox('Y-Axis', all_cols, key='area_y')
        fig = px.area(df, x=x_col, y=y_col, title=f'Area Chart of {y_col} over {x_col}')

    elif graph_type == 'Violin Plot':
        st.subheader("Violin Plot Options")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox('X-Axis', all_cols, key='violin_x')
        with col2:
            y_col = st.selectbox('Y-Axis', all_cols, key='violin_y')
        fig = px.violin(df, x=x_col, y=y_col, title=f'Violin Plot of {y_col} by {x_col}')

    elif graph_type == 'Strip Plot':
        st.subheader("Strip Plot Options")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox('X-Axis', all_cols, key='strip_x')
        with col2:
            y_col = st.selectbox('Y-Axis', all_cols, key='strip_y')
        fig = px.strip(df, x=x_col, y=y_col, title=f'Strip Plot of {y_col} by {x_col}')

    # Display the plot if one was created
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Download Options ---
        st.sidebar.header("💾 Download Plot")
        try:
            img_bytes = fig.to_image(format="png", scale=2)
            st.sidebar.download_button(
                label="🖼️ Download as PNG",
                data=img_bytes,
                file_name=f"{graph_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        except RuntimeError:
            st.sidebar.warning("PNG export is unavailable on this server.")
        
        html_bytes = fig.to_html()
        st.sidebar.download_button(
            label="🌐 Download as HTML",
            data=html_bytes,
            file_name=f"{graph_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )




# --- Helper Functions (ML Modeling) ---

# config.py content
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

DATASET_COLUMNS = {
    "Boston Housing": {
        "old_cols": ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"],
        "new_cols": [
            "Crime Rate", "Zoned Land", "Industrial Proportion", "River Proximity", 
            "NOX Concentration", "Rooms per Dwelling", "Age of Property", 
            "Distance to Employment Centers", "Highway Accessibility", "Property Tax Rate", 
            "Pupil-Teacher Ratio", "Black Population Proportion", "Lower Status Population", 
            "Median Value"
        ],
        "target_col": "Median Value"
    },
    "California Housing": {
        "old_cols": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "medianHouseValue"],
        "new_cols": [
            "Median Income", "House Age", "Average Rooms", "Average Bedrooms", 
            "Population", "Average Occupancy", "Latitude", "Longitude", "Median House Value"
        ],
        "target_col": "Median House Value"
    },
    "Medical Insurance Costs": {
        "old_cols": ["age", "sex", "bmi", "children", "smoker", "region", "charges"],
        "new_cols": ["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Charges"],
        "target_col": "Charges"
    },
    "Fish Market": {
        "old_cols": ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"],
        "new_cols": ["Species", "Weight", "Vertical Length", "Diagonal Length", "Cross Length", "Height", "Width"],
        "target_col": "Weight"
    },
    "Salary Data": {
        "old_cols": ["YearsExperience", "Salary"],
        "new_cols": ["Years of Experience", "Salary"],
        "target_col": "Salary"
    }
}

# ml.py content
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics

def train_model(X, y, missing_strategy, outlier_strategy, categorical_cols, model_type, alpha, cv_folds):
    # One-hot encode categorical columns
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Handle missing values
    if missing_strategy == 'Drop Rows':
        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data[y.name]
    elif missing_strategy == 'Mean Imputation':
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
    elif missing_strategy == 'Median Imputation':
        X = X.fillna(X.median())
        y = y.fillna(y.median())

    # Handle outliers
    if outlier_strategy == 'Remove Outliers (IQR)':
        # Combine X and y for consistent outlier removal
        data = pd.concat([X, y], axis=1)
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        X = data[X.columns]
        y = data[y.name]

    # Select model based on user choice
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'Lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("Invalid model type selected.")

    # Perform Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, X_test, y_test, cv_scores

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics_dict = {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "r2_score": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "mse": metrics.mean_squared_error(y_test, y_pred)
    }
    return metrics_dict

def create_plots(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Predicted vs Actual Values Plot with Regression Line
    fig_actual_vs_predicted = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Predicted vs Actual Values', trendline="ols")
    fig_actual_vs_predicted.add_shape(
        type='line',
        x0=y_test.min(),
        y0=y_test.min(),
        x1=y_test.max(),
        y1=y_test.max(),
        line=dict(color='red', dash='dash')
    )

    # Residual Plot
    fig_residuals = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title='Residual Plot')
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

    return {
        "actual_vs_predicted": fig_actual_vs_predicted,
        "residuals": fig_residuals
    }

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Data Explorer", "ML Modeler"])

# --- Centralized Data Upload and Sample Dataset Selection ---
sample_datasets = {
    "None": None,
    "Boston Housing": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "California Housing": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    "Medical Insurance Costs": "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
    "Fish Market": "https://raw.githubusercontent.com/Ankit152/Fish-Market/main/Fish.csv",
    "Salary Data": "https://raw.githubusercontent.com/saikrishnapotluri/salary_Data.csv/master/Salary_Data.csv"
}

with st.sidebar:
    st.markdown("---")
    st.header("Data Source")
    selected_dataset = st.selectbox("Or choose a sample dataset", list(sample_datasets.keys()), key="central_selected_dataset")
    st.session_state.selected_dataset = selected_dataset # Store in session state
    uploaded_file_obj = st.file_uploader("Choose a CSV file", type="csv", key="central_file_uploader")

    # Add delimiter and encoding options here
    st.subheader("CSV Parsing Options")
    delimiter = st.selectbox("CSV Delimiter", (',', ';', '\t'), key="central_delimiter_select")
    encoding = st.selectbox("Encoding", ('utf-8', 'latin1', 'iso-8859-1'), key="central_encoding_select")

    source_file = None
    if selected_dataset != "None":
        source_file = sample_datasets[selected_dataset]
    elif uploaded_file_obj is not None:
        source_file = uploaded_file_obj

    if source_file is not None:
        try:
            if isinstance(source_file, str): # It's a URL from sample datasets
                df = pd.read_csv(source_file, delimiter=delimiter, encoding=encoding)
            else: # It's an uploaded file object
                df = pd.read_csv(source_file, delimiter=delimiter, encoding=encoding)
            st.session_state.df = df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}. Please check your file and parsing options.")
            st.session_state.df = None
    else:
        st.session_state.df = None

# --- Conditional Content Display ---
if selection == "Data Explorer":
    if st.session_state.df is not None:
        st.markdown('<h1 style="font-weight:bold;color:#1e3a8a;">InsightiGraph</h1>', unsafe_allow_html=True)
        
        # --- Download CSV ---
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        

        # --- Main Application Tabs ---
        tab1, tab2 = st.tabs(["📊 Data Overview", "🎨 Visualization Studio"])

        with tab1:
            display_data_overview(st.session_state.df)
        with tab2:
            display_visualizations(st.session_state.df)

    else:
        display_welcome_message()
elif selection == "ML Modeler":
    st.title("Linear Regression Model Trainer 📈")
    st.write("Welcome! This app helps you train a Linear Regression model with customizable preprocessing steps. Follow the steps below to get started:")

    # Use the centralized DataFrame
    df = st.session_state.df

    if df is not None:
        # Add a reset button
        if st.button("Reset App 🔄", key="reset_button"):
            for key in ['feature_cols', 'target_col']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # The missing_strategy and outlier_strategy are still needed for the ML model training
        # So, we need to define them here, but they are not tied to re-reading the CSV
        missing_strategy = st.selectbox("Missing Values", ('Drop Rows', 'Mean Imputation', 'Median Imputation'), key="missing_strategy_select")
        outlier_strategy = st.selectbox("Outlier Handling", ('None', 'Remove Outliers (IQR)'), key="outlier_strategy_select")

        try:
            
            # Improve column names for Boston Housing dataset
            # We need to check if selected_dataset is not None and use that for column mapping
            if st.session_state.selected_dataset != "None" and st.session_state.selected_dataset in DATASET_COLUMNS:
                dataset_config = DATASET_COLUMNS[st.session_state.selected_dataset]
                df.columns = dataset_config["new_cols"]
                target_col_name = dataset_config["target_col"]
                
                # Reorder columns to place target column at the end
                cols = df.columns.tolist()
                if target_col_name in cols:
                    cols.remove(target_col_name)
                    cols.append(target_col_name)
                    df = df[cols]
                
                # Pre-populate target selection
                st.session_state.target_col = cols[-1]

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

            st.write("### Data Preview 📊")
            st.write(df.head())

            with st.expander("Explore Data Visualizations (Optional) 📊"):
                selected_outlier_cols = st.multiselect("Select columns for Outlier Visualizations (Box Plots) 📈", numeric_cols)
                selected_distribution_cols = st.multiselect("Select columns for Feature Distribution Plots (Histograms) 📈", numeric_cols)
                show_correlation_viz = st.checkbox("Show Correlation Matrix Heatmap 📈")

                if selected_outlier_cols:
                    st.write("### Outlier Visualizations 📊")
                    for col in selected_outlier_cols:
                        fig = px.box(df, y=col, title=f'Box Plot of {col}', height=400, width=600)
                        st.plotly_chart(fig)

                if selected_distribution_cols:
                    st.write("### Feature Distributions 📊")
                    for col in selected_distribution_cols:
                        fig = px.histogram(df, x=col, title=f'Distribution of {col}', height=400, width=600)
                        st.plotly_chart(fig)

                if show_correlation_viz:
                    st.write("### Correlation Matrix Heatmap 📈")
                    corr_matrix = df[numeric_cols].corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix", height=600, width=800)
                    st.plotly_chart(fig_corr)

            # Step 2: Feature & Target Selection
            st.header("2. Select Features and Target 🎯")

            if df.empty:
                st.warning("The uploaded file is empty.")
            elif len(df.columns) == 0:
                st.warning("The uploaded file has no columns.")
            else:
                if not numeric_cols and not categorical_cols:
                    st.warning("No numeric or categorical columns found in the dataset.")
                else:
                    if not numeric_cols:
                        st.warning("No numeric columns found. Linear Regression requires numeric features.")

                    selected_categorical_cols = []
                    if categorical_cols:
                        with st.expander("Categorical Feature Handling 🗂️"):
                            selected_categorical_cols = st.multiselect("Select categorical columns for One-Hot Encoding", categorical_cols)

                    target_col = st.selectbox("Select target column (y) (Numeric Only)", numeric_cols, index=numeric_cols.index(st.session_state.get('target_col')) if st.session_state.get('target_col') in numeric_cols else 0)
                    feature_cols = st.multiselect("Select feature columns (X) (Numeric and One-Hot Encoded Categorical)", [col for col in numeric_cols if col != target_col] + selected_categorical_cols)

                    # Step 3: Train Model
                    st.header("3. Train Linear Regression Model 🧠")

                    with st.expander("Model Selection & Hyperparameters ⚙️", expanded=True):
                        model_type = st.selectbox("Select Model Type", ('Linear Regression', 'Ridge', 'Lasso'), key="model_type_select")
                        alpha = None
                        if model_type in ['Ridge', 'Lasso']:
                            alpha = st.slider(f"Alpha for {model_type}", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                        
                        

                    if st.button("Train Model ✨", key="train_button"):
                        if not feature_cols:
                            st.warning("⚠️ Please select at least one feature column.")
                        elif not target_col:
                            st.warning("⚠️ Please select a target column.")
                        elif target_col in feature_cols:
                            st.warning("⚠️ Target column cannot be a feature column. Please select a different target or remove it from features.")
                        else:
                            X = df[feature_cols]
                            y = df[target_col]

                            with st.spinner("Training model... This might take a moment! ⏳"):
                                model, X_test, y_test, cv_scores = train_model(X, y, missing_strategy, outlier_strategy, selected_categorical_cols, model_type, alpha, CV_FOLDS)
                            metrics = get_metrics(model, X_test, y_test)
                            
                            # Step 4: Display Model Performance
                            st.header("4. Model Performance 📊")
                            st.write("Here are the key performance indicators for your trained model:")
                            
                            metrics_df = pd.DataFrame({
                                "Metric": ["Coefficients", "Intercept", "R² Score", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Cross-Validation R² (Mean)", "Cross-Validation R² (Std)"],
                                "Value": [
                                    f"{metrics['coef']:.4f}" if isinstance(metrics['coef'], (int, float)) else f"[{', '.join([f'{c:.4f}' for c in metrics['coef']])}]",
                                    f"{metrics['intercept']:.4f}",
                                    f"{metrics['r2_score']:.4f}",
                                    f"{metrics['mae']:.4f}",
                                    f"{metrics['mse']:.4f}",
                                    f"{cv_scores.mean():.4f}",
                                    f"{cv_scores.std():.4f}"
                                ]
                            })
                            st.dataframe(metrics_df, hide_index=True)

                            # Step 5: Visualizations
                            st.header("5. Visualizations 📈")
                            st.write("Explore the relationship between predicted and actual values:")
                            plots = create_plots(model, X_test, y_test)
                            st.plotly_chart(plots['actual_vs_predicted'], use_container_width=True)

                            st.write("### Residual Plot 📉")
                            st.plotly_chart(plots['residuals'], use_container_width=True)

                            st.write("### Feature Importance (Coefficients) 📊")
                            if hasattr(model, 'coef_') and len(feature_cols) > 0:
                                coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
                                coef_df['Absolute Coefficient'] = coef_df['Coefficient'].abs()
                                coef_df = coef_df.sort_values(by='Absolute Coefficient', ascending=False)
                                fig_coef = px.bar(coef_df, x='Feature', y='Coefficient', title='Feature Importance', height=400, width=600)
                                st.plotly_chart(fig_coef)
                            else:
                                st.info("Coefficients are not available for this model type or no features were selected.")

                            # Step 6: Download Options
                            st.header("6. Download Options 💾")
                            
                            # Download Trained Model
                            import pickle
                            model_bytes = pickle.dumps(model)
                            st.download_button(
                                label="Download Trained Model (.pkl)",
                                data=model_bytes,
                                file_name="linear_regression_model.pkl",
                                mime="application/octet-stream"
                            )

                            # Download Predictions
                            y_pred = model.predict(X_test)
                            predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                            csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions (.csv)",
                                data=csv_predictions,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )

        except Exception as e:
            st.error(f"An error occurred: {e} 🐞 Please check your data and selections.")
    else:
        st.info("Please upload a dataset or select a sample dataset from the sidebar to start ML Modeling.")
