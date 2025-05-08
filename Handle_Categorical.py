import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import traceback
import time

def handle_categorical(data):
    try:
        st.header("🏷️ Handle Categorical Columns")
        
        # Initialize session state variables if they don't exist
        if 'current_data' not in st.session_state:
            st.session_state['current_data'] = data.copy()
        
        if 'original_data' not in st.session_state:
            st.session_state['original_data'] = data.copy()
            
        if 'processed_categorical_columns' not in st.session_state:
            st.session_state['processed_categorical_columns'] = set()
            
        if 'categorical_transformations' not in st.session_state:
            st.session_state['categorical_transformations'] = {}

        # Work with the current state of the data
        df = st.session_state['current_data'].copy()

        # Get categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        remaining_columns = [col for col in categorical_columns if col not in st.session_state['processed_categorical_columns']]

        if len(categorical_columns) == 0:
            st.warning("No categorical columns found in the dataset.")
            return df

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Categorical Columns", len(categorical_columns))
        with col2:
            st.metric("Processed Columns", len(st.session_state['processed_categorical_columns']))
        with col3:
            st.metric("Remaining Columns", len(remaining_columns))
            
        st.subheader("Categorical Columns")
        st.table(categorical_columns.tolist())

        if len(remaining_columns) == 0:
            st.success("All categorical columns have been processed!")
            return df

        # Let the user choose a column
        selected_column = st.selectbox(
            "Select a categorical column to handle",
            remaining_columns
        )

        # Provide options for handling the column
        method = st.radio(
            "Choose a method to handle the column:",
            options=["One-Hot Encoding", "Label Encoding", "Keep Original"],
            index=0,
            help="One-Hot Encoding: Creates binary columns for each category\n"
                 "Label Encoding: Converts categories to numerical values\n"
                 "Keep Original: Maintains the original categorical format"
        )

        # Preview the transformation
        transformed_df = df.copy()
        if method == "One-Hot Encoding":
            transformed_df = pd.get_dummies(transformed_df, columns=[selected_column])
            new_cols = [col for col in transformed_df.columns if col.startswith(f"{selected_column}_")]
            st.write("### Preview of One-Hot Encoded Columns")
            st.dataframe(transformed_df[new_cols].head())
        elif method == "Label Encoding":
            label_encoder = LabelEncoder()
            transformed_df[f"{selected_column}_encoded"] = label_encoder.fit_transform(transformed_df[selected_column])
            st.write("### Preview of Label Encoded Column")
            st.dataframe(pd.DataFrame({
                'Original': transformed_df[selected_column],
                'Encoded': transformed_df[f"{selected_column}_encoded"]
            }).drop_duplicates())

        st.write("### Preview of Complete Dataset")
        st.dataframe(transformed_df.head())

        if st.button("Apply Changes", type="primary"):
            try:
                # Store transformation in session state
                st.session_state['categorical_transformations'][selected_column] = method
                
                # Add column to processed set
                st.session_state['processed_categorical_columns'].add(selected_column)
                
                # Update the current data in session state
                st.session_state['current_data'] = transformed_df
                
                # Show transformation history
                st.write("### Transformation History")
                history_df = pd.DataFrame(
                    list(st.session_state['categorical_transformations'].items()),
                    columns=['Column', 'Transformation']
                )
                st.dataframe(history_df)
                
                st.success(f"Successfully applied {method} to column '{selected_column}'")
                
                # Show current state of the data
                st.write("### Current State of Dataset")
                st.dataframe(st.session_state['current_data'].head())
                st.session_state.df =  st.session_state['current_data'] 
                
                # Return the transformed dataframe directly
                return transformed_df
                              
            except Exception as e:
                st.warning(f"Error applying transformation")
                return df

        # Add a reset button
        if st.button("Reset All Transformations"):
            st.session_state['current_data'] = st.session_state['original_data'].copy()
            st.session_state['processed_categorical_columns'] = set()
            st.session_state['categorical_transformations'] = {}
            st.success("All transformations have been reset!")
            st.rerun()

        # Return the current state of the data
        return df

    except Exception as e:
        st.warning(f"An error occurred while handling categorical columns")
        return data