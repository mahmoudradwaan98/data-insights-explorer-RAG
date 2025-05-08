import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
import requests
import subprocess
from Handle_Duplicates import handle_duplicates
from Handle_Missing import handle_missing_values
from Handle_Outliers import handle_outliers
from Column_Operations import rename_columns, remove_columns, convert_column_types
from Visualizations import show_correlation_analysis, visualize
from Handle_Categorical import handle_categorical
from Show_Unique import show_unique_values
from io import StringIO
import io

class CustomStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.container = st.empty()
    
    def on_llm_start(self, *args, **kwargs):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.tokens.append(token)
        # Join all tokens and display them
        complete_text = "".join(self.tokens)
        self.container.markdown(complete_text)
    
    def on_llm_end(self, *args, **kwargs):
        pass
    
    def on_llm_error(self, error: str, *args, **kwargs):
        st.warning(f"Error: {error}")

SUPPORTED_EXTENSIONS = ['csv', 'xlsx', 'xls']


# Initialize session states
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Cache the Ollama model initialization
@st.cache_resource
def initialize_ollama():
    """Initialize and cache the Ollama model"""
    return Ollama(
        model="llama3.2:1b",
        temperature=0.7,
        num_ctx=2048
    )

# Cache the conversation chain creation
@st.cache_resource
def create_conversation_chain():
    """Create and cache the conversation chain"""
    if st.session_state.conversation_chain is None:
        llm = initialize_ollama()
        template = """
        You are a helpful assistant that answers questions about the dataset.
       
        Human Question: {human_input}
        
        Assistant: Let me help you with that question about your dataset.
        """
        prompt = PromptTemplate(
            input_variables=["human_input"],
            template=template
        )
        st.session_state.conversation_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False
        )
    return st.session_state.conversation_chain

# Helper functions with caching
@st.cache_data
def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

@st.cache_data
def check_model_available(model_name="llama3.2:1b"):
    """Check if the specified model is available in Ollama"""
    try:
        response = requests.get(f"http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            # Check if the specific model is available
            return any(model["name"] == model_name for model in available_models)
    except requests.exceptions.ConnectionError:
        st.warning("Unable to connect to Ollama. Ensure Ollama is running.")
        return False
    return False

def pull_model(model_name="llama3.2:1b"):
    """Pull the specified model using Ollama CLI"""
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        return True
    except subprocess.CalledProcessError:
        st.warning(f"Error pulling model: subprocess error")
        return False
    except FileNotFoundError:
        st.warning(f"Error pulling model: ollama command not found")
        return False
    except Exception as e:
        st.warning(f"Error pulling model: {str(e)}")
        return False

@st.cache_data
def load_data_file(file):
    try:
        # Get file extension
        file_extension = file.name.split('.')[-1].lower()
        
        # Validate file extension
        if file_extension not in SUPPORTED_EXTENSIONS:
            st.warning(f"❌ Unsupported file format: '{file_extension}'")
            st.markdown("""
            ### Supported File Formats
            - CSV (.csv)
            - Excel (.xlsx, .xls)
            
            ℹ️ Please upload a file with one of these extensions.
            """)
            st.stop()
        
        # Load based on file type
        if file_extension == 'csv':
            # Try different encodings
            encodings_to_try = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']
            
            for encoding in encodings_to_try:
                try:
                    csv_file = StringIO(file.getvalue().decode(encoding))
                    return pd.read_csv(csv_file)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.warning(f"Error reading CSV with {encoding} encoding: {str(e)}")
                    continue
            
            st.warning("Could not read the CSV file with any supported encoding")
            return None
            
        elif file_extension in ['xlsx', 'xls']:
            try:
                # Use engine='openpyxl' for .xlsx, 'xlrd' for .xls
                engine = 'openpyxl' if file_extension == 'xlsx' else 'xlrd'
                return pd.read_excel(file, engine=engine)
            except ImportError as ie:
                st.warning(f"Required package not installed: {str(ie)}. Please install {engine} package.")
                return None
            except Exception as e:
                st.warning(f"Error reading Excel file")
                st.info("Tips: Make sure the file is not corrupted and has proper permissions.")
                return None
    except Exception as e:
        st.warning(f"Error processing file: {str(e)}")
        st.markdown("""
        ### 🚨 File Upload Error
        - Check file integrity
        - Ensure file is not corrupted
        - Verify file format
        - Try saving the CSV file with UTF-8 encoding
        """)
        return None

def download_data(df, file_base_name,file_extension):
    st.header("Download Your Dataset")
    try:
        modified_file_name = f"{file_base_name}_modified.{file_extension}"
        if file_extension.lower() == 'csv':
            csv = df.to_csv(index=False)
            download_button = st.download_button(
                label="Download CSV 📥",
                data=csv,
                file_name=modified_file_name,
                mime="text/csv",
                key='downloaded',
            )
                 
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Sheet1")
            excel_data = output.getvalue()
            
            download_button = st.download_button(
                label="Download Excel 📥",
                data=excel_data,
                file_name=modified_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='downloaded',
            )    
    except Exception as e:
        st.session_state.downloaded = False 
        st.warning(f"Sorry, the download failed. Please try again later. Error: {str(e)}")

def reset_all_flags():
    keys_to_reset = [
        'rename_columns_clicked',
        'remove_columns_clicked',
        'convert_types_clicked',
        'visualize_button_clicked',
        'correlation_clicked',
        'missing_button_clicked',
        'unique_values_button_clicked',
        'categorical_button_clicked',
        'outliers_button_clicked',
        'duplicates_button_clicked'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
           st.session_state[key] = False

@st.cache_data
def get_column_details(df):
    column_details = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        non_null_count = df[column].count()
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        unique_values = df[column].nunique()
        sample_values = ", ".join(map(str, df[column].dropna().unique()[:5]))
        
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val = df[column].min()
            max_val = df[column].max()
            mean_val = df[column].mean()
            stats = f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}"
        else:
            stats = "N/A"
        
        column_details.append({
            "Column Name": column,
            "Data Type": dtype,
            "Non-Null Count": f"{non_null_count} ({(non_null_count/len(df))*100:.1f}%)",
            "Null Count": f"{null_count} ({null_percentage:.1f}%)",
            "Unique Values": unique_values,
            "Sample Values": sample_values,
            "Statistics": stats
        })
    return pd.DataFrame(column_details)
@st.cache_data
def get_data_context(df):
    if df is None:
        return "No data available"
    
    # Basic dataset overview
    context = f"Dataset Information:\n"
    context += f"- Total Rows: {len(df)}\n"
    context += f"- Total Columns: {len(df.columns)}\n\n"
    
    # Column details
    context += "Columns and Their Characteristics:\n"
    for column in df.columns:
        # Determine column type
        if pd.api.types.is_numeric_dtype(df[column]):
            context += f"- {column} (Numeric):\n"
            context += f"  * Min: {df[column].min()}\n"
            context += f"  * Max: {df[column].max()}\n"
            context += f"  * Mean: {df[column].mean():.2f}\n"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            context += f"- {column} (Date/Time):\n"
            context += f"  * Earliest: {df[column].min()}\n"
            context += f"  * Latest: {df[column].max()}\n"
        else:
            context += f"- {column} (Categorical/Text):\n"
            context += f"  * Unique Values: {df[column].nunique()}\n"
        
        # Check for missing values
        missing = df[column].isnull().sum()
        if missing > 0:
            context += f"  * Missing Values: {missing} ({missing/len(df)*100:.2f}%)\n"
    
    # Sample data preview
    context += "\nSample Data (First 3 rows):\n"
    context += df.head(3).to_string()
    
    return context
st.markdown("""
<style>
    .chat-container {
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .bot-message {
        background-color: #2E2E2E;
        color: #FFFFFF;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Check Ollama status
if not check_ollama_running():
    st.warning("""
    Ollama is not running! Please follow these steps:
    1. Install Ollama from https://ollama.ai
    2. Open a terminal/command prompt
    3. Run the command: ollama serve
    4. Refresh this page
    """)
    st.stop()
# Check if model is available
if not check_model_available():
    st.warning("Attempting to pull the llama3.2:1b instruct model...")
    if pull_model(model_name="llama3.2:1b"):
        st.success("Successfully pulled llama3.2:1b model!")
    else:
        st.warning(""" 
        Failed to pull llama3.2:1b model. Please try manually:
        1. Open a terminal/command prompt
        2. Run: ollama pull llama3.2:1b
        3. Refresh this page
        """)
        st.stop()
# Main interface
st.title("💬 Chat with your dataset ")
# Sidebar for file upload
with st.sidebar:
    st.header("📂 Upload Your Dataset")
    
    # Enhanced file uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_EXTENSIONS,
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS).upper()}"
    )
        
# Main processing logic
if uploaded_file is not None:
    try:
        
        # Load the data with validation
        if st.session_state.df is None:
            st.session_state.df = load_data_file(uploaded_file)
            st.session_state.file_extension = uploaded_file.name.split('.')[-1].lower()
            st.session_state.file_base_name = ".".join(uploaded_file.name.split('.')[:-1])
            
        # Initialize all state flags if not present
        flag_keys = [
            'rename_columns_clicked',
            'remove_columns_clicked',
            'convert_types_clicked',
            'visualize_button_clicked',
            'correlation_clicked',
            'outliers_button_clicked',
            'unique_values_button_clicked',
            'categorical_button_clicked',
            'duplicates_button_clicked',
            'missing_button_clicked',
            'downloaded'
        ]
        
        for key in flag_keys:
            if key not in st.session_state:
                st.session_state[key] = False

        # Validate successful data load
        if st.session_state.df is not None:
            # File details
            st.sidebar.success("✅ File uploaded successfully!")
            
            # Detailed file information
            file_details = {
                "Filename": uploaded_file.name,
                "File type": uploaded_file.name.split('.')[-1].upper(),
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.sidebar.header("File Details")
            # Create a more visually appealing details display
            for key, value in file_details.items():
                st.sidebar.markdown(f"**{key}**: {value}")
            
            # Sidebar buttons for data exploration
            st.sidebar.header("Data Exploration")
            show_info_button = st.sidebar.button("Dataset Info", use_container_width=True)
            show_describe_button = st.sidebar.button("Statistical Summary", use_container_width=True)
            unique_button = st.sidebar.button("Unique Values", use_container_width=True)
            rename_columns_button = st.sidebar.button("Rename Columns", use_container_width=True)
            remove_columns_button = st.sidebar.button("Remove Columns", use_container_width=True)
            convert_types_button = st.sidebar.button("Convert Column Types", use_container_width=True)
            visualize_button = st.sidebar.button("Visualize Columns", use_container_width=True)
            correlation_button = st.sidebar.button("Correlation Analysis", use_container_width=True)
            categorical_button = st.sidebar.button("Handle Categorical Columns", use_container_width=True)
            handle_duplicates_button = st.sidebar.button("Handle Duplicates", use_container_width=True)
            handle_missing_values_button = st.sidebar.button("Handle Missing Values", use_container_width=True)
            handle_outliers_button = st.sidebar.button("Handle Outliers", use_container_width=True)
            download_data_button = st.sidebar.button("Download Data", use_container_width=True)
            if handle_duplicates_button:
                reset_all_flags()
                st.session_state.duplicates_button_clicked = True
            if handle_missing_values_button:
                reset_all_flags()
                st.session_state.missing_button_clicked = True
            if handle_outliers_button or  st.session_state.outliers_button_clicked:
                reset_all_flags()
                st.session_state.outliers_button_clicked = True
            if download_data_button:
                reset_all_flags()
                download_data(st.session_state.df,st.session_state.file_base_name,st.session_state.file_extension)
            if show_info_button:
                reset_all_flags()
                column_info_df = get_column_details(st.session_state.df)
                st.dataframe(column_info_df)
            if rename_columns_button:
                reset_all_flags() 
                st.session_state.rename_columns_clicked = True
            if remove_columns_button:
                reset_all_flags() 
                st.session_state.remove_columns_clicked = True
            if convert_types_button:
                reset_all_flags()
                st.session_state.convert_types_clicked = True
            if visualize_button:
                reset_all_flags()
                st.session_state.visualize_button_clicked = True
            if correlation_button:
                reset_all_flags()
                st.session_state.correlation_clicked = True
            if categorical_button:
                reset_all_flags()
                st.session_state.categorical_button_clicked = True
            if unique_button: 
               reset_all_flags()
               st.session_state.unique_values_button_clicked = True
            if show_describe_button:
                reset_all_flags()
                with st.expander("📈 Comprehensive Statistical Summary", expanded=True):
                    desc_stats = st.session_state.df.describe(include='all')
                    st.dataframe(
                        desc_stats.round(2),
                        use_container_width=True
                    )
                    st.markdown("### 🔍 Quick Insights")
                    total_rows = len(st.session_state.df)
                    total_columns = len(st.session_state.df.columns)
                    numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
                    categorical_columns = st.session_state.df.select_dtypes(include=['object']).columns
                    cols = st.columns(4)
                    metrics = [
                        (f"Total Rows", total_rows),
                        (f"Total Columns", total_columns),
                        (f"Numeric Columns", len(numeric_columns)),
                        (f"Categorical Columns", len(categorical_columns))
                    ]
                    
                    for col, (label, value) in zip(cols, metrics):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                            <h4>{label}</h4>
                            <p style="font-size: 24px; margin: 0;">{value}</p>
                            </div>
                            """, unsafe_allow_html=True)
            if st.session_state.missing_button_clicked:
                st.session_state.df = handle_missing_values(st.session_state.df)
            if st.session_state.outliers_button_clicked:
                st.session_state.df = handle_outliers(st.session_state.df)
            if st.session_state.duplicates_button_clicked:
                st.session_state.df=handle_duplicates(st.session_state.df)
            if st.session_state.rename_columns_clicked:
                rename_columns(st.session_state.df)
            if st.session_state.remove_columns_clicked:
                remove_columns(st.session_state.df)
            if st.session_state.convert_types_clicked:
                convert_column_types(st.session_state.df)
            if st.session_state.visualize_button_clicked:
                visualize(st.session_state.df)          
            if st.session_state.correlation_clicked:
                show_correlation_analysis(st.session_state.df)
            if st.session_state.categorical_button_clicked:
                handle_categorical(st.session_state.df)
            if st.session_state.unique_values_button_clicked:
                show_unique_values(st.session_state.df)
            if st.session_state.downloaded:
                st.success("Dataset downloaded successfully!")
                st.session_state.downloaded = False
        # Initialize conversation chain if not already initialized
        if st.session_state.conversation_chain is None:
            st.session_state.conversation_chain = create_conversation_chain()
        
        # Data preview section
        with st.expander("📊 Preview Dataset", expanded=False):
            st.dataframe(st.session_state.df.head())    
      
        # Chat input area
        with st.container():
            user_input = st.text_input("Ask a question about your data:", 
                                        placeholder="e.g., What are the columns in this dataset?")
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                send_button = st.button("Send 📤")
            with col2:
                clear_button = st.button("Clear 🗑️")

        # Handle clear chat
        if clear_button:
            st.session_state.chat_history = []

        # Handle send message
        if send_button and user_input:
            try:
                # Get cached data context
                data_context = get_data_context(st.session_state.df)
                                
# Create a container for the streaming response
                response_container = st.container()
                
                # Create custom callback handler
                callback_handler = CustomStreamlitCallbackHandler()
                chain = LLMChain(
                    llm=initialize_ollama(),
                    prompt=PromptTemplate(
                            input_variables=["data_context", "human_input"],
                            template="""
                            You are a helpful assistant that answers questions about the dataset.
                            Dataset Context:
                            {data_context}
                            Human Question: {human_input}
                        Assistant: Let me help you with that question about your dataset.
                        """
                    ),
                    verbose= False
                )
                
                st.session_state.is_streaming = True
                
                with st.spinner("Processing..."):
                    response = chain.predict(
                        data_context=data_context,
                        human_input=user_input,
                        callbacks=[callback_handler]
                    )
                
                st.session_state.chat_history.append({
                    "user": user_input, 
                    "bot": response
                })
                st.session_state.is_streaming = False
                
            except Exception as e:
                st.warning(f"Error: {str(e)}")
                
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                # User message
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="user-message">{message["user"]}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Bot message
                st.markdown(f"""
                    <div class="chat-container">
                        <div class="bot-message">{message["bot"]}</div>
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Error processing file: {str(e)}")
else:
  st.markdown("""
### 🚀 Welcome to Data Insights Explorer!

Unlock powerful insights from your data through natural conversation and interactive analysis.

#### 📋 Quick Start Guide:
1. **Upload Your Data**  
   - 📁 Supports CSV and Excel files  
   - 🔄 Automatic data validation and preview  

2. **Chat with Your Data**  
   - 💬 Ask questions in natural language  
   - 🤖 Receive AI-powered insights  

#### 🛠️ Advanced Features:
- **Data Quality Analysis**  
  - Column-wise statistics  
  - Missing value detection  
  - Outlier identification  

- **Interactive Processing** 🧹  
  - Handle missing values  
  - Remove duplicates  
  - Treat outliers  

- **AI Capabilities**  
  - Natural language queries  
  - Automated insights  

#### 💡 Example Questions to Ask:
- What are the main trends in my data?  
- Show me the distribution of values in column X  
- Identify potential correlations between variables  
- Summarize the key statistics of my dataset  

---

**Ready to explore your data? Upload a file to get started!** 📈
""")

        
