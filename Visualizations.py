import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

def show_correlation_analysis(df):
    try:
        st.header("Correlation Analysis")
        st.session_state.correlation_clicked = True
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns to perform correlation analysis.")
            return

        # Let user select specific columns
        try:
            selected_columns = st.multiselect(
                "Select columns for correlation analysis:",
                numeric_cols,
                default=list(numeric_cols)[:2],
                help="Choose at least 2 columns to analyze their correlation"
            )
        except Exception as e:
            st.error(f"Error selecting columns: {str(e)}")
            return

        if len(selected_columns) < 2:
            st.warning("Please select at least 2 columns for correlation analysis.")
            return

        try:
            # Visualization options
            viz_type = st.radio(
                "Select Visualization Type:",
                ["Heatmap", "Correlation Matrix", "Line Plot"],
                horizontal=True,
                help="Choose different visualization methods to explore correlations"
            )

            if viz_type == "Heatmap":
                try:
                    st.subheader("Correlation Heatmap")
                    
                    # Heatmap customization
                    col1, col2 = st.columns(2)
                    with col1:
                        cmap_option = st.selectbox(
                            "Color Scheme",
                            ["coolwarm", "viridis", "RdBu", "seismic", "YlOrRd"],
                            help="Choose the color scheme for the heatmap"
                        )
                    with col2:
                        annotate = st.checkbox("Show Values", value=True)

                    fig = plt.figure(figsize=(12, 8))
                    correlation_matrix = df[selected_columns].corr()
                    
                    if len(selected_columns) > 2:
                        mask = np.triu(np.ones_like(correlation_matrix))
                    else:
                        mask = None
                        
                    sns.heatmap(
                        correlation_matrix,
                        annot=annotate,
                        cmap=cmap_option,
                        mask=mask,
                        vmin=-1,
                        vmax=1,
                        center=0,
                        fmt='.2f'
                    )
                    plt.title("Correlation Heatmap")
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")

            elif viz_type == "Correlation Matrix":
                try:
                    st.subheader("Interactive Correlation Matrix")
                    corr_matrix = df[selected_columns].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        zmin=-1,
                        zmax=1,
                        text=np.round(corr_matrix, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorscale='RdBu'
                    ))
                    
                    fig.update_layout(
                        title="Interactive Correlation Matrix",
                        height=600,
                        width=800
                    )
                    
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error creating correlation matrix: {str(e)}")

            elif viz_type == "Line Plot":
                try:
                    st.subheader("Correlation Line Plot")
                    
                    if len(selected_columns) > 1:
                        normalized_data = df[selected_columns].apply(
                            lambda x: (x - x.min()) / (x.max() - x.min())
                        )
                        
                        fig = go.Figure()
                        for col in selected_columns:
                            fig.add_trace(go.Scatter(
                                y=normalized_data[col],
                                name=col,
                                mode='lines'
                            ))
                        
                        fig.update_layout(
                            title="Normalized Values Comparison",
                            xaxis_title="Index",
                            yaxis_title="Normalized Value",
                            template="simple_white"
                        )
                        
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error creating line plot: {str(e)}")

            # Show correlation coefficients
            try:
                if len(selected_columns) >= 2:
                    st.subheader("Correlation Coefficients")
                    corr_df = df[selected_columns].corr()
                    
                    def style_correlation(val):
                        color = 'red' if val < 0 else 'green'
                        return f'color: {color}'
                    
                    styled_corr = corr_df.style.format('{:.3f}').applymap(style_correlation)
                    st.dataframe(styled_corr, use_container_width=True)

                    # Correlation Interpretation
                    st.subheader("Correlation Interpretation")
                    for i in range(len(selected_columns)):
                        for j in range(i+1, len(selected_columns)):
                            try:
                                col1, col2 = selected_columns[i], selected_columns[j]
                                corr_value = corr_df.loc[col1, col2]
                                
                                # Determine correlation strength
                                if abs(corr_value) >= 0.7:
                                    strength = "Strong"
                                    color = "#FF4B4B" if corr_value < 0 else "#2ECC71"
                                elif abs(corr_value) >= 0.4:
                                    strength = "Moderate"
                                    color = "#FF8C4B" if corr_value < 0 else "#82E0AA"
                                else:
                                    strength = "Weak"
                                    color = "#FFB74B" if corr_value < 0 else "#D5F5E3"

                                st.markdown(f"""
                                <div style="padding: 10px; border-left: 5px solid {color}; 
                                          margin: 10px 0; background-color: #f0f2f6;">
                                    <strong>{col1}</strong> and <strong>{col2}</strong>: 
                                    {strength} {corr_value < 0 and 'negative' or 'positive'} 
                                    correlation ({corr_value:.3f})
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error interpreting correlation for {col1} and {col2}: {str(e)}")
            except Exception as e:
                st.error(f"Error calculating correlation coefficients: {str(e)}")

        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred during correlation analysis: {str(e)}")
        return
                
                
                
                
def visualize(data):
    st.title("Dynamic Chart Visualizer")
    columns = data.columns.tolist()
    x_column = st.selectbox("Select X-axis Column", columns)
    y_column = st.selectbox("Select Y-axis Column (if applicable)", [None] + columns)

    chart_type = st.selectbox(
        "Select Chart Type", [
            "Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Boxplot", "Pie Chart"]
    )

    if st.button("Generate Chart"):
        fig = None

        if chart_type == "Scatter Plot":
            fig = px.scatter(data, x=x_column, y=y_column, title=f"{chart_type} of {y_column} vs {x_column}")
        elif chart_type == "Line Plot":
           if y_column:
                aggregated_data = data.groupby(x_column)[y_column].sum().reset_index()
                fig = px.line(aggregated_data, x=x_column, y=y_column, title=f"{chart_type} of Total {y_column} vs {x_column}")
           else:
                st.warning("Y-axis column is required for Line Plot.")
        elif chart_type == "Bar Plot":
            if y_column:
                aggregated_data = data.groupby(x_column, as_index=False)[y_column].sum()
                sorted_data = aggregated_data.sort_values(by=y_column, ascending=False).head(10)
                fig = px.bar(sorted_data, x=x_column, y=y_column, title=f"{chart_type} of Top 10 {x_column} by Total {y_column}")
            else:
                counts = data[x_column].value_counts().head(10).reset_index()
                counts.columns = [x_column, 'Count']
                counts = counts.sort_values(by='Count', ascending=False)
                fig = px.bar(counts, x=x_column, y='Count', title=f"{chart_type} of Top 10 {x_column} by Count")
        elif chart_type == "Histogram":
            fig = px.histogram(data, x=x_column, title=f"{chart_type} of {x_column}", nbins=20, marginal="box")
        elif chart_type == "Boxplot":
            if y_column:
                fig = px.box(data, x=x_column, y=y_column, title=f"{chart_type} of {y_column} grouped by {x_column}")
            else:
                fig = px.box(data, y=x_column, title=f"{chart_type} of {x_column}")
        elif chart_type == "Pie Chart":
            if y_column:
                aggregated_data = data.groupby(x_column, as_index=False)[y_column].sum()
                sorted_data = aggregated_data.sort_values(by=y_column, ascending=False).head(10)
                fig = px.pie(sorted_data, names=x_column, values=y_column, title=f"{chart_type} of Top 10 {x_column} by Total {y_column}")
            else:
                counts = data[x_column].value_counts().head(10).reset_index()
                counts.columns = [x_column, 'Count']
                counts = counts.sort_values(by='Count', ascending=False)
                fig = px.pie(counts, names=x_column, values='Count', title=f"{chart_type} of Top 10 {x_column} by Count")
        else:
            st.warning("Unsupported chart type!")

        if fig:
            st.plotly_chart(fig, use_container_width=True)