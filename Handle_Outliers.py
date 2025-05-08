import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import traceback

def calculate_outlier_bounds(series, multiplier=1.5):
    """Calculate outlier bounds using IQR method with adjustable multiplier"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return lower_bound, upper_bound

def find_columns_with_outliers(dataframe, numeric_columns, iqr_multiplier=1.5):
    """Enhanced outlier detection with detailed statistics"""
    columns_with_outliers = []
    outlier_details = {}
    outlier_stats = {}
    
    for column in numeric_columns:
        # Calculate bounds
        lower_bound, upper_bound = calculate_outlier_bounds(dataframe[column], iqr_multiplier)
        
        # Identify outliers
        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        
        if len(outliers) > 0:
            columns_with_outliers.append(column)
            outlier_details[column] = outliers[column]
            
            # Calculate detailed statistics
            stats = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(dataframe)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'low_outliers': len(dataframe[dataframe[column] < lower_bound]),
                'high_outliers': len(dataframe[dataframe[column] > upper_bound])
            }
            outlier_stats[column] = stats
    
    return columns_with_outliers, outlier_details, outlier_stats

def handle_outliers(data):
    try:
        st.title("Handle Outliers")
        st.session_state.outliers_button_clicked = True

        # Store original data for comparison
        original_data = data.copy()

        # Select only numeric columns
        numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_columns:
            st.warning("No numerical columns found in the dataset!")
            return data

        # Initial outlier detection with selected multiplier
        columns_with_outliers, outlier_details, outlier_stats = find_columns_with_outliers(
            data, numeric_columns
        )

        if not columns_with_outliers:
            st.success("No outliers detected in any columns.")
            return data

        # Manage columns with outliers in session state
        if 'columns_with_outliers' not in st.session_state:
            st.session_state.columns_with_outliers = columns_with_outliers

        if not st.session_state.columns_with_outliers:
            st.success("All outliers have been handled in all columns.")
            return data

        # Show outlier summary for all columns
        st.subheader("Outlier Summary")
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Total Columns with Outliers", len(columns_with_outliers))
        with summary_cols[1]:
            total_outliers = sum(stats['outlier_count'] for stats in outlier_stats.values())
            st.metric("Total Outliers", total_outliers)
        with summary_cols[2]:
            avg_percentage = np.mean([stats['outlier_percentage'] for stats in outlier_stats.values()])
            st.metric("Average Outlier Percentage", f"{avg_percentage:.2f}%")

        # Column selection with outlier stats
        selected_column = st.selectbox(
            "Select a numeric column with outliers",
            options=st.session_state.columns_with_outliers,
            format_func=lambda x: f"{x} ({outlier_stats[x]['outlier_count']} outliers, {outlier_stats[x]['outlier_percentage']:.1f}%)"
        )

        # Enhanced method selection with descriptions
        method_descriptions = {
            "Remove Outliers": "Remove rows containing outliers (impacts dataset size)",
            "Replace with Mean": "Replace outliers with column mean (preserves dataset size)",
            "Replace with Median": "Replace outliers with column median (preserves dataset size)",
            "Clip Values": "Cap outliers at the boundary values (preserves dataset size)"
        }
        
        method = st.selectbox(
            "Select a method to handle outliers",
            options=list(method_descriptions.keys()),
            format_func=lambda x: f"{x} - {method_descriptions[x]}"
        )

        # Display detailed analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### Outlier Bounds")
            bounds_data = {
                "Metric": ["IQR", "Lower Bound", "Upper Bound"],
                "Value": [
                    outlier_stats[selected_column]['upper_bound'] - outlier_stats[selected_column]['lower_bound'],
                    outlier_stats[selected_column]['lower_bound'],
                    outlier_stats[selected_column]['upper_bound']
                ]
            }
            st.dataframe(pd.DataFrame(bounds_data))
            st.write("### Statistical Analysis")
            stats_df = data[[selected_column]].describe()
            st.dataframe(stats_df)

        with col2:
            st.write("### Current Outliers")
            outliers_df = outlier_details[selected_column].to_frame()
            st.dataframe(outliers_df)
            st.write(f"Found {len(outliers_df)} outliers:")
            st.write(f"- Below lower bound: {outlier_stats[selected_column]['low_outliers']}")
            st.write(f"- Above upper bound: {outlier_stats[selected_column]['high_outliers']}")

        with col3:
            st.write("### Preview After Handling")
            modified_data = data[selected_column].copy()
            lower_bound = outlier_stats[selected_column]['lower_bound']
            upper_bound = outlier_stats[selected_column]['upper_bound']

            if method == "Remove Outliers":
                modified_data = modified_data[(modified_data >= lower_bound) & (modified_data <= upper_bound)]
            elif method == "Replace with Mean":
                mean_value = modified_data.mean()
                modified_data = modified_data.mask(
                    (modified_data < lower_bound) | (modified_data > upper_bound),
                    mean_value
                )
            elif method == "Replace with Median":
                median_value = modified_data.median()
                modified_data = modified_data.mask(
                    (modified_data < lower_bound) | (modified_data > upper_bound),
                    median_value
                )
            elif method == "Clip Values":
                modified_data = modified_data.clip(lower=lower_bound, upper=upper_bound)

            st.dataframe(modified_data.to_frame())

        # Create box plot to visualize distribution
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=data[selected_column],
            name='Original',
            boxpoints='outliers'
        ))
        fig.add_trace(go.Box(
            y=modified_data,
            name='After Handling',
            boxpoints='outliers'
        ))
        fig.update_layout(title=f'Distribution Comparison for {selected_column}')
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Apply Changes", type="primary"):
            original_count = len(data)

            # Apply the selected outlier handling method
            if method == "Remove Outliers":
                data = data[(data[selected_column] >= lower_bound) & (data[selected_column] <= upper_bound)]
            elif method == "Replace with Mean":
                mean_value = data[selected_column].mean()
                data.loc[(data[selected_column] < lower_bound) | (data[selected_column] > upper_bound), selected_column] = mean_value
            elif method == "Replace with Median":
                median_value = data[selected_column].median()
                data.loc[(data[selected_column] < lower_bound) | (data[selected_column] > upper_bound), selected_column] = median_value
            elif method == "Clip Values":
                data[selected_column] = data[selected_column].clip(lower=lower_bound, upper=upper_bound)

            # Show changes
            new_count = len(data)
            st.write(f"Rows before: **{original_count}**, Rows after: **{new_count}**")
            
            # Show before/after comparison
            comparison = pd.DataFrame({
                'Original': original_data[selected_column].describe(),
                'After Handling': data[selected_column].describe()
            })
            st.write("### Before/After Statistics")
            st.dataframe(comparison)
   
            # Remove the handled column from columns_with_outliers
            st.session_state.columns_with_outliers.remove(selected_column)

            # If no more columns with outliers, reset the state
            if not st.session_state.columns_with_outliers:
                st.success("All outliers have been handled successfully!")
                del st.session_state.columns_with_outliers
                return data

     

        return data

    except Exception as e:
        st.warning(f"Handling outliers failed")
        return data