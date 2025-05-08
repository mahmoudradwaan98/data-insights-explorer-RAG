import streamlit as st

def handle_duplicates(data):
    # Ensure we're working with a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Count duplicates
    duplicate_count = df.duplicated().sum()
    
    # If no duplicates, show a warning and return original data
    if duplicate_count == 0:
        st.warning('No Duplicate Rows Found')
        return df
    
    # Ensure method is in session state
    if 'duplicate_method' not in st.session_state:
        st.session_state.duplicate_method = "Keep First"
    
    # Main display and interaction
    st.write(f"Number of Duplicate Rows: {duplicate_count}")
    
    # Method selection with session state preservation
    duplicate_method = st.selectbox(
        "Select a method to handle duplicates",
        ["Keep First", "Keep Last", "Drop All"],
        index=["Keep First", "Keep Last", "Drop All"].index(st.session_state.duplicate_method),
        key="duplicate_method_selector"
    )
    
    # Update session state method
    st.session_state.duplicate_method = duplicate_method
    
    # Dataset preview before handling
    st.write("### Dataset Preview (Before Handling)")
    st.dataframe(df.head(10))
    
    # Apply button
    if st.button("Apply Duplicate Handling", type="primary"):
        try:
            # Store original row count
            original_count = len(df)
            if duplicate_method == "Keep First":
                df.drop_duplicates(keep="first", inplace=True)
                st.success("First occurrences of duplicates have been kept.")
            elif duplicate_method == "Keep Last":
                df.drop_duplicates(keep="last", inplace=True)
                st.success("Last occurrences of duplicates have been kept.")
            elif duplicate_method == "Drop All":
                df.drop_duplicates(keep=False, inplace=True)
                st.success("All duplicates have been dropped.")
            new_count = len(df)
            st.write(f"Rows before: **{original_count}**, Rows after: **{new_count}**")
            st.write("### Dataset Preview (After Handling)")
            st.dataframe(df.head(10))
        except Exception as e:
            st.warning(f"Error processing duplicates")
            st.info("Please check your data and try again.")
    return df