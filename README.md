# Data Insights Explorer

Welcome to the Data Insights Explorer! This project allows you to upload datasets and interact with them through natural language queries using an AI-powered assistant Using RAG!. It provides various tools for data preprocessing, exploration, and visualization.

## Features

- **Upload and Preview Data**: Supports CSV and Excel files with automatic data validation.
- **Interactive Chat**: Ask questions about your dataset and get AI-powered responses.
- **Data Preprocessing**:
  - Handle missing values, outliers, duplicates, and perform column operations.
  - Handle categorical columns with One-Hot Encoding or Label Encoding.
  - Analyze columns with unique values and visualize their distributions.
- **Data Visualization**: Generate various charts and perform correlation analysis.
- **Download Processed Data**: Download the modified dataset after processing.

## Installation

To run this project, you need to have Python and the required libraries installed. Follow these steps to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/emansa3ed/data-insights-explorer.git
    cd data-insights-explorer
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r Requirements.txt
    ```

## Usage

1. **Start the Streamlit Application**:
    ```bash
    streamlit run App.py
    ```

2. **Upload Your Dataset**: Use the sidebar to upload a CSV or Excel file.

3. **Interact with Your Data**:
    - **Chat**: Use the chat interface to ask questions about your dataset.
    - **Explore**: Use the sidebar buttons to view dataset info, statistical summaries, unique values, and more.
    - **Preprocess**: Handle missing values, outliers, duplicates, and perform column operations.
    - **Visualize**: Generate scatter plots, line plots, bar plots, histograms, boxplots, and pie charts.

4. **Download Processed Data**: After processing, download the modified dataset using the provided button.

## Data Preprocessing Features

### Handle Categorical Columns

The `handle_categorical` function allows you to handle categorical columns using the following methods:
- **One-Hot Encoding**: Creates binary columns for each category.
- **Label Encoding**: Converts categories to numerical values.
- **Keep Original**: Maintains the original categorical format.

### Handle Duplicates

The `handle_duplicates` function provides methods to manage duplicate rows:
- **Keep First**: Keep the first occurrence of duplicate rows.
- **Keep Last**: Keep the last occurrence of duplicate rows.
- **Drop All**: Remove all duplicate rows.

### Handle Missing Values

The `handle_missing_values` function provides methods to handle missing values in the dataset:
- **Fill with Mean**: Fill missing values with the mean of the column.
- **Fill with Median**: Fill missing values with the median of the column.
- **Fill with Mode**: Fill missing values with the mode of the column.
- **Drop Rows**: Remove rows containing missing values.
- **Random Imputation**: Randomly impute missing values with existing values in the column.

### Handle Outliers

The `handle_outliers` function provides methods to handle outliers in numerical columns:
- **Remove Outliers**: Remove rows containing outliers.
- **Replace with Mean**: Replace outliers with the column mean.
- **Replace with Median**: Replace outliers with the column median.
- **Clip Values**: Cap outliers at the boundary values.

### Show Unique Values

The `show_unique_values` function allows you to analyze columns with fewer than 50 unique values, providing:
- **Distribution**: View value counts and percentages.
- **Details**: Detailed information including missing values and basic statistics.
- **Visualization**: Visualize the distribution of values using histograms or bar charts.

## Project Structure

- `app.py`: Main Streamlit application script.
- `requirements.txt`: List of required Python packages.
- `Handle_Duplicates.py`: Module for handling duplicate values.
- `Handle_Missing.py`: Module for handling missing values.
- `Handle_Outliers.py`: Module for handling outliers.
- `Column_Operations.py`: Module for renaming, removing, and converting column types.
- `Visualizations.py`: Module for data visualization.
- `Handle_Categorical.py`: Module for handling categorical data.
- `Show_Unique.py`: Module for showing unique values in columns.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## Acknowledgements

- Thanks to the [Streamlit](https://www.streamlit.io/) team for their great framework.
- Special thanks to the developers of [LangChain](https://github.com/langchain-ai/langchain).

---

Happy Exploring! 🚀
