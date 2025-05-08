import streamlit as st
import pandas as pd
import random
import time

def random_imputation(column):
    """Randomly impute missing values with existing values in the column."""
    non_null_values = column.dropna().values
    return column.apply(lambda x: random.choice(non_null_values) if pd.isnull(x) else x)

def handle_missing_values(data):
    st.title("Handle Missing Values")
    st.session_state.missing_button_clicked = True

    # Check for missing values
    missing_counts = data.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()

    # If no missing values
    if not columns_with_missing:
        st.success("✅ No missing values found in the dataset!")
        return data

    # Initialize session state for tracking
    if 'missing_handling_state' not in st.session_state:
        st.session_state.missing_handling_state = {
            'selected_column': columns_with_missing[0],
            'missing_method': "Fill with Mode",
            'original_data': data[columns_with_missing[0]].copy(),
            'modified_data': None,
        }

    # Get the current column and method
    selected_column = st.session_state.missing_handling_state['selected_column']
    method = st.session_state.missing_handling_state.get('missing_method', "Fill with Mode")

    # Column selection
    selected_column = st.selectbox(
        "Select a column with missing values",
        columns_with_missing,
        index=columns_with_missing.index(selected_column)
    )
    st.session_state.missing_handling_state['selected_column'] = selected_column

    # Determine available methods based on column data type
    column_dtype = data[selected_column].dtype
    if pd.api.types.is_numeric_dtype(column_dtype):
        available_methods = ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Random Imputation"]
    else:
        available_methods = ["Fill with Mode", "Random Imputation", "Drop Rows"]

    # Method selection
    method = st.selectbox(
        "Select a method to handle missing values",
        available_methods,
        index=available_methods.index(method)
    )
    st.session_state.missing_handling_state['missing_method'] = method

    # Create columns for before and after views
    col1, col2 = st.columns(2)

    with col1:
        st.write("Before Handling")
        # Show original data with missing values
        original_data = data[selected_column]
        st.dataframe(original_data.head(10))

        # Show missing value count
        missing_count = original_data.isnull().sum()
        st.metric("Missing Values", missing_count)

    with col2:
        st.write("Preview After Handling")
        # Prepare modified data for preview
        try:
            modified_data = original_data.copy()

            if method == "Fill with Mean" and pd.api.types.is_numeric_dtype(original_data):
                modified_data.fillna(original_data.mean(), inplace=True)
            elif method == "Fill with Median" and pd.api.types.is_numeric_dtype(original_data):
                modified_data.fillna(original_data.median(), inplace=True)
            elif method == "Fill with Mode":
                modified_data.fillna(original_data.mode()[0], inplace=True)
            elif method == "Random Imputation":
                modified_data = random_imputation(original_data)
            elif method == "Drop Rows":
                modified_data.dropna(inplace=True)

            st.dataframe(modified_data.head(10))

            # Show new missing value count
            missing_count_after = modified_data.isnull().sum()
            st.metric("Missing Values", missing_count_after)

        except Exception as e:
            st.warning(f"Error previewing changes")

    # Apply changes button
    if st.button("Apply Changes", type="primary"):
        try:
            # Apply the selected method to the entire dataframe
            if method == "Fill with Mean" and pd.api.types.is_numeric_dtype(data[selected_column]):
                data[selected_column].fillna(data[selected_column].mean(), inplace=True)
            elif method == "Fill with Median" and pd.api.types.is_numeric_dtype(data[selected_column]):
                data[selected_column].fillna(data[selected_column].median(), inplace=True)
            elif method == "Fill with Mode":
                data[selected_column].fillna(data[selected_column].mode()[0], inplace=True)
            elif method == "Random Imputation":
                data[selected_column] = random_imputation(data[selected_column])
            elif method == "Drop Rows":
                data.dropna(subset=[selected_column], inplace=True)

            st.success(f"✅ Changes applied successfully to '{selected_column}' using '{method}' method!")

            # Update session state to show the next column with missing values
            missing_counts = data.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0].index.tolist()

            if columns_with_missing:
                # Update to the next column
                next_column = columns_with_missing[0]
                st.session_state.missing_handling_state.update({
                    'selected_column': next_column,
                    'original_data': data[next_column].copy(),
                    'modified_data': None
                })
            else:
                # Clear session state if no missing values remain
                st.success("✅ All missing values have been handled!")
                del st.session_state.missing_handling_state
            time.sleep(2)
            # Force rerun to immediately reflect the updated state
            st.rerun()

        except Exception as e:
            st.warning(f"Error applying changes while handling missing values")

    return data
