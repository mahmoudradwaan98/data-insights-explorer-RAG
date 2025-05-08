import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import traceback

def show_unique_values(df):
    try:
        st.header("Unique Values Analysis")
        
        # Initialize session state for page persistence
        if 'selected_unique_column' not in st.session_state:
            st.session_state['selected_unique_column'] = None
        
        # Filter columns with less than 50 unique values
        filtered_columns = []
        column_unique_counts = {}
        for column in df.columns:
            unique_count = df[column].nunique()
            if unique_count < 50:
                filtered_columns.append(column)
                column_unique_counts[column] = unique_count
        
        if not filtered_columns:
            st.warning("No columns found with fewer than 50 unique values.")
            return
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Columns", len(df.columns))
        with col2:
            st.metric("Columns with <50 Unique Values", len(filtered_columns))
        
        # Create selectbox for column selection with session state
        selected_column = st.selectbox(
            "Select a column to analyze:",
            filtered_columns,
            format_func=lambda x: f"{x} ({column_unique_counts[x]} unique values)",  # Fixed the extra curly brace
            key='unique_values_selectbox',
            index=filtered_columns.index(st.session_state['selected_unique_column']) if st.session_state['selected_unique_column'] in filtered_columns else 0
        )
        
        # Update session state
        st.session_state['selected_unique_column'] = selected_column
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📋 Details", "📈 Visualization"])
        
        with tab1:
            # Calculate value counts and percentages
            value_counts = df[selected_column].value_counts()
            value_percentages = df[selected_column].value_counts(normalize=True).round(4) * 100
            
            # Combine counts and percentages
            distribution_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percentages
            })
            distribution_df = distribution_df.round(2)
            
            st.write("### Value Distribution")
            st.dataframe(distribution_df)
            
            # Display basic statistics
            missing_values = df[selected_column].isnull().sum()
            st.metric("Missing Values", missing_values)
        
        with tab2:
            st.write("### Detailed Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Column Details:")
                details = {
                    "Data Type": df[selected_column].dtype,
                    "Unique Values": column_unique_counts[selected_column],
                    "Missing Values": missing_values,
                    "Missing Percentage": round((missing_values / len(df)) * 100, 2)
                }
                st.table(pd.Series(details, name="Value"))
            
            with col2:
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    st.write("Numeric Statistics:")
                    stats = {
                        "Mean": df[selected_column].mean(),
                        "Median": df[selected_column].median(),
                        "Std Dev": df[selected_column].std(),
                        "Min": df[selected_column].min(),
                        "Max": df[selected_column].max()
                    }
                    st.table(pd.Series(stats, name="Value").round(2))
        
        with tab3:
            st.write("### Visualization")
            
            try:
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    fig = px.histogram(df, x=selected_column, 
                                     title=f'Distribution of {selected_column}',
                                     marginal='box')
                else:
                    # For categorical data, create a bar chart
                    value_counts = df[selected_column].value_counts()
                    fig = px.bar(x=value_counts.index, 
                                y=value_counts.values,
                                title=f'Distribution of {selected_column}',
                                labels={'x': selected_column, 'y': 'Count'})
                    fig.update_layout(showlegend=False)
                
                st.plotly_chart(fig)
                
            except Exception as e:
                st.warning(f"Error creating visualization")
        

    except Exception as e:
        st.warning(f"An error occurred in show_unique_values")
      