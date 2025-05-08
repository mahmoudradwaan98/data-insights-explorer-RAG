import streamlit as st
import pandas as pd
import time

def rename_columns(df):
    try:
        st.header("Rename Columns")
        
        # Create a selectbox for column selection
        selected_column = st.selectbox(
            "Choose column to rename",
            options=df.columns,
            key="column_selector"
        )

        # Only show input field if a column is selected
        if selected_column:
            # Create input for new name with current name as initial value
            new_name = st.text_input(
                "Enter new column name",
                value=selected_column,
                key=f"new_name_input_{selected_column}"
            )

            # Add Apply button
            if st.button("Apply Rename", type="primary"):
                # Validation checks
                if not new_name.strip():
                    st.warning("⚠️ Please enter a valid column name.")
                    return
                
                # Check for duplicate names (except if it's the same column)
                if new_name in df.columns and new_name != selected_column:
                    st.warning("⚠️ This column name already exists. Duplicate names are not allowed.")
                    return
                
                # If validation passes, rename the column
                if new_name != selected_column:
                    try:
                        # Create copy of DataFrame with renamed column
                        st.session_state.df = df.rename(columns={selected_column: new_name})
                        st.success(f"✅ Column '{selected_column}' successfully renamed to '{new_name}'!")
                        
                        # Clear the input
                        if f"new_name_input_{selected_column}" in st.session_state:
                            del st.session_state[f"new_name_input_{selected_column}"]
                        
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Error renaming column: {selected_column}")
                else:
                    st.info("No change in column name.")

    except Exception as e:
        st.warning(f"Renaming column {selected_column} failed")
        return

def remove_columns(df):
    st.header("Remove Columns")
    st.session_state.remove_columns_clicked = True
    INITIAL_DISPLAY_COUNT = 15
    total_columns = len(df.columns)
    st.markdown("""
    <style>
    .column-container {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        
    }
    
    .column-item { 
        padding: 8px 15px;
        margin: 5px 0;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: background-color 0.3s, transform 0.2s;
        border: 1px solid #000;
    }
    .column-item:hover {
     border: 1px solid #000;
    transform: scale(1.02);
    }
    .column-number {
        background-color: #D3D3D3;
        color: #000000;
        padding: 2px 8px;
        border-radius: 3px;
        margin-right: 8px;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for confirmation
    if 'column_to_delete' not in st.session_state:
        st.session_state.column_to_delete = None
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("<div class='column-container'>", unsafe_allow_html=True)
        if total_columns > INITIAL_DISPLAY_COUNT:
            st.info(f"Showing {INITIAL_DISPLAY_COUNT} of {total_columns} columns")
            show_all = st.checkbox("Show all columns", key="show_all_columns")
            display_columns = df.columns if show_all else df.columns[:INITIAL_DISPLAY_COUNT]
        else:
            display_columns = df.columns
        st.markdown("### Available Columns")
        for i, column in enumerate(display_columns):
            col_left, col_right = st.columns([4, 1])
            with col_left:
                st.markdown(f"""
                    <div class='column-item'>
                        <span>
                            <span class='column-number'>{i + 1}</span>
                            {column}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            with col_right:
                if st.button("🗑️", key=f"delete_{column}", help=f"Delete column '{column}'"):
                    st.session_state.column_to_delete = column
                    st.session_state.confirm_delete = True
                    st.rerun()
        if total_columns > INITIAL_DISPLAY_COUNT and not show_all:
            st.markdown(f"<i>... and {total_columns - INITIAL_DISPLAY_COUNT} more columns</i>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)   
            

        st.markdown("</div>", unsafe_allow_html=True)

    # Show confirmation dialog
    if st.session_state.confirm_delete and st.session_state.column_to_delete:
        with st.form(key='confirm_delete_form'):
            st.warning(f"⚠️ Are you sure you want to delete the column '{st.session_state.column_to_delete}'?")
            st.write("This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            with col1:
                confirm = st.form_submit_button("Yes, Delete", type="primary")
            with col2:
                cancel = st.form_submit_button("No, Cancel")

            if confirm:
                try:
                    # Remove the column
                    st.session_state.df = df.drop(columns=[st.session_state.column_to_delete])
                    st.success(f"✅ Column '{st.session_state.column_to_delete}' has been removed successfully!")
                    
                    # Reset states
                    st.session_state.column_to_delete = None
                    st.session_state.confirm_delete = False
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.warning(f"Error removing column: {st.session_state.column_to_delete}")
            
            if cancel:
                # Reset states
                st.session_state.column_to_delete = None
                st.session_state.confirm_delete = False
                st.rerun()

def convert_column_types(df):
    try:
        st.header("Convert Column Types")
        st.session_state.convert_types_clicked = True

        # Style definitions remain the same...

        # Create column selector with current types
        columns_with_types = [f"{col} (Current: {df[col].dtype})" for col in df.columns]
        selected_column_with_type = st.selectbox(
            "Select column to convert",
            options=columns_with_types,
            help="Choose a column to change its data type"
        )

        # Extract column name from selection
        selected_column = selected_column_with_type.split(" (Current:")[0]
        current_type = str(df[selected_column].dtype)

        # Check number of unique values for the selected column
        num_unique_values = df[selected_column].nunique()

        # Define available data types with descriptions
        TYPE_MAPPINGS = {
            'numeric': {
                'int64': 'Integer (whole numbers)',
                'float64': 'Float (decimal numbers)',
            },
            'text': {
                'string': 'Text',
                'category': 'Category (for repeated text values)',
            },
            'datetime': {
                'datetime64[ns]': 'Date and Time',
            },
            'boolean': {
                'boolean': 'True/False values'
            }
        }

        # Create radio buttons for type categories
        type_category = st.radio(
            "Select type category",
            list(TYPE_MAPPINGS.keys()),
            horizontal=True
        )

        # Show specific types based on category
        new_type = st.selectbox(
            "Select specific type",
            options=list(TYPE_MAPPINGS[type_category].keys()),
            format_func=lambda x: TYPE_MAPPINGS[type_category][x]
        )

        # Check if boolean conversion is allowed
        boolean_conversion_allowed = num_unique_values == 2

        # Show warning if boolean conversion is not allowed
        if new_type == 'boolean' and not boolean_conversion_allowed:
            st.warning(f"Converting to boolean is not allowed for this column. The column must have exactly 2 unique values, but it has {num_unique_values} unique values.")
            return

        try:
            # Show preview section
            st.markdown("### Preview Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='preview-container'>", unsafe_allow_html=True)
                st.markdown("**Before Conversion:**")
                st.write(df[selected_column].head())
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='preview-container'>", unsafe_allow_html=True)
                st.markdown("**After Conversion (Preview):**")
                try:
                    if new_type == 'datetime64[ns]':
                        preview_data = pd.to_datetime(df[selected_column].head(), errors='coerce')
                    elif new_type == 'boolean':
                        if boolean_conversion_allowed:
                            preview_data = df[selected_column].head().astype(bool)
                        else:
                            raise ValueError("Boolean conversion not allowed")
                    else:
                        preview_data = df[selected_column].head().astype(new_type)
                    st.write(preview_data)
                    conversion_possible = True
                except Exception as e:
                    st.warning("This conversion is not possible. Please select a different data type.")
                    conversion_possible = False
                st.markdown("</div>", unsafe_allow_html=True)

            # Show warning for potential data loss
            if new_type == 'int64' and current_type == 'float64':
                st.warning("⚠️ Converting to integer might lose decimal values.")

            # Add conversion button only if conversion is possible
            if conversion_possible:
                if st.button("Apply Conversion", type="primary"):
                    try:
                        # Perform the conversion
                        if new_type == 'datetime64[ns]':
                            st.session_state.df[selected_column] = pd.to_datetime(df[selected_column], errors='coerce')
                        elif new_type == 'boolean':
                            if boolean_conversion_allowed:
                                st.session_state.df[selected_column] = df[selected_column].astype(bool)
                            else:
                                raise ValueError("Boolean conversion not allowed")
                        else:
                            st.session_state.df[selected_column] = df[selected_column].astype(new_type)

                        st.success(f"✅ Successfully converted '{selected_column}' to {new_type}")
                        time.sleep(1.5)
                        st.rerun()

                    except Exception as e:
                        st.warning(f"Error converting '{selected_column}' to {new_type}")

        except Exception as e:
            st.warning("This conversion is not possible. Please select a different data type.")

    except Exception as e:
        st.warning(f"An error occurred: {str(e)}")
        return
