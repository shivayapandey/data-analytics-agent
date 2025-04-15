import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from langchain.agents import AgentType
import json
import textwrap
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import AzureChatOpenAI
try:
    import pandasql as psql
    SQL_AVAILABLE = True
except ImportError:
    st.warning("SQL functionality disabled: pandasql module not available. Install with 'pip install pandasql'")
    SQL_AVAILABLE = False
from PIL import Image

AZURE_DEPLOYMENT = st.secrets["AZURE_DEPLOYMENT"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]

os.environ["AZURE_API_KEY"] = AZURE_API_KEY

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'example_questions' not in st.session_state:
    st.session_state.example_questions = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'mode' not in st.session_state:
    st.session_state.mode = "normal"
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = None
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

def nl_to_sql(question, df):
    """Convert natural language query to SQL query using LLM"""
    try:
        llm = AzureChatOpenAI(
            openai_api_version=AZURE_API_VERSION,
            azure_deployment=AZURE_DEPLOYMENT,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.2,
        )
        
        col_descriptions = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            col_descriptions.append(f"- {col} ({dtype}): Example value: {sample}")
        
        table_info = "\n".join(col_descriptions)
        
        prompt = f"""Given a DataFrame with the following columns:
        
        {table_info}
        
        Convert the following natural language query to a valid SQL query:
        "{question}"
        
        The table name is 'df'. Return ONLY the SQL query without any explanation, markdown formatting, or quotes.
        The query should be executable using the pandasql library.
        """
        
        response = llm.predict(prompt).strip()
        sql_query = response
        if sql_query.startswith('"') and sql_query.endswith('"'):
            sql_query = sql_query[1:-1]
        if sql_query.startswith("'") and sql_query.endswith("'"):
            sql_query = sql_query[1:-1]
        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql", 1)[1]
        if sql_query.startswith("```"):
            sql_query = sql_query.split("```", 1)[1]
        if sql_query.endswith("```"):
            sql_query = sql_query.rsplit("```", 1)[0]
            
        return sql_query.strip()
    except Exception as e:
        st.session_state.error_log.append(f"nl_to_sql error: {str(e)}")
        st.error("Unable to generate SQL query. Please try a simpler question.")
        return None

def execute_sql_query(query, df):
    """Execute SQL query on the dataframe"""
    try:
        result = psql.sqldf(query, locals())
        return result
    except Exception as e:
        st.session_state.error_log.append(f"execute_sql_query error: {str(e)}")
        st.error("Error executing SQL query. Please check your question.")
        return None

def execute_visualization_code(code, df):
    """Execute visualization code and return the plot"""
    try:
        exec_globals = {
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'df': df
        }
        
        buf = io.BytesIO()
        
        exec(code, exec_globals)
        
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()  
        
        buf.seek(0)
        test_img = Image.open(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.session_state.error_log.append(f"execute_visualization_code error: {str(e)}")
        st.error(f"Failed to generate visualization: {str(e)}")
        return None
def generate_visualization_code(df, query):
    """Generate Python code for visualization using Matplotlib/Seaborn"""
    try:
        llm = AzureChatOpenAI(
            openai_api_version=AZURE_API_VERSION,
            azure_deployment=AZURE_DEPLOYMENT,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.2,
        )
        
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else 0
            }
        
        prompt = f"""Generate ONLY Python code using Matplotlib or Seaborn to visualize: "{query}"

Dataset columns: {columns}
Column types: {dtypes}
Numeric columns stats: {stats}

REQUIREMENTS:
1. Return ONLY executable Python code with NO explanations or markdown.
2. Use only this method to save the figure: plt.savefig('/tmp/visualization.png', format='png', dpi=100, bbox_inches='tight')
3. Do not use plt.show() or any display functions.
4. Process a DataFrame available as 'df'.
5. Use plt.figure(figsize=(12, 8)) at the beginning.
6. Use plt.tight_layout() before saving.
7. Handle possible missing values in the data.
8. Use only the provided columns; do not assume other columns exist.
"""
        
        response = llm.predict(prompt)
        code = response.strip()
        if code.startswith("```python"):
            code = code.split("```python", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        unsafe_patterns = ["os.system", "open(", "exec(", "eval("]
        if any(pattern in code for pattern in unsafe_patterns):
            raise ValueError("Unsafe code detected")
        
        return code.strip()
    except Exception as e:
        st.session_state.error_log.append(f"generate_visualization_code error: {str(e)}")
        st.error("Unable to generate visualization. Please try a different description.")
        return None

def generate_example_questions(df):
    """Generate example questions based on the first 3 rows"""
    try:
        llm = AzureChatOpenAI(
            openai_api_version=AZURE_API_VERSION,
            azure_deployment=AZURE_DEPLOYMENT,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            temperature=0.7,
        )
        columns = df.columns.tolist()
        sample_data = df.head(3).to_dict()
        prompt = f"""Based on this dataset with columns {columns} and sample data (first 3 rows):
{json.dumps(sample_data, indent=2)}

Generate 4 complete analytical questions that would be interesting to ask about this data.
Return them in a JSON array format like this:

{{"questions": [
    {{"text": "question 1", "button_text": "question 1"}},
    {{"text": "question 2", "button_text": "question 2"}},
    ...
]}}

Make questions specific to this dataset's content and structure.
Use the full question text for both text and button_text fields.
"""
        response = llm.predict(prompt)
        try:
            questions = json.loads(response)['questions']
            return questions
        except json.JSONDecodeError:
            default_questions = [
                {"text": f"What is the distribution of values in {columns[0]}?", "button_text": f"What is the distribution of values in {columns[0]}?"},
                {"text": f"How does {columns[0]} correlate with {columns[1]}?", "button_text": f"How does {columns[0]} correlate with {columns[1]}?"},
                {"text": "What are the key statistical patterns in the dataset?", "button_text": "What are the key statistical patterns in the dataset?"},
                {"text": "Can you provide insights from the dataset?", "button_text": "Can you provide insights from the dataset?"}
            ]
            return default_questions[:4]
    except Exception as e:
        st.session_state.error_log.append(f"generate_example_questions error: {str(e)}")
        return [
            {"text": "What are the key statistical patterns in the dataset?", "button_text": "What are the key statistical patterns in the dataset?"},
            {"text": "Can you provide insights from the dataset?", "button_text": "Can you provide insights from the dataset?"},
            {"text": "How is the data distributed?", "button_text": "How is the data distributed?"},
            {"text": "What are the main trends in the dataset?", "button_text": "What are the main trends in the dataset?"}
        ]

def process_question(question):
    """Process a question or visualization request"""
    if st.session_state.agent is None:
        return
    
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Processing..."):
            # Check if the query is a visualization request
            viz_keywords = ["plot", "chart", "graph", "visualize", "diagram"]
            is_viz = any(keyword in question.lower() for keyword in viz_keywords)
            
            if is_viz and st.session_state.mode == "normal":
                viz_code = generate_visualization_code(st.session_state.df, question)
                if viz_code:
                    with st.expander("View Generated Code"):
                        st.code(viz_code, language="python")
                    
                    viz_result = execute_visualization_code(viz_code, st.session_state.df)
                    if viz_result:
                        st.image(viz_result)
                        viz_result.seek(0)
                        b64 = base64.b64encode(viz_result.read()).decode()
                        href = f'<a href="data:image/png;base64,{b64}" download="visualization.png" class="download-btn">Download Visualization</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.markdown("""
                        <style>
                        .download-btn {
                            display: inline-block;
                            padding: 0.5em 1em;
                            background-color: #4CAF50;
                            color: white;
                            text-decoration: none;
                            border-radius: 4px;
                            margin-top: 10px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        message_placeholder.markdown("Visualization generated successfully.")
                    else:
                        message_placeholder.markdown("Failed to generate visualization. Please try a different description.")
                else:
                    message_placeholder.markdown("Unable to generate visualization code.")
            else:
                # Handle as a question-answering task
                response = st.session_state.agent.run(question)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

st.set_page_config(page_title="Data Analytics Assistant", layout="wide")

st.title("Data Analytics Assistant")

# Query type selector
st.sidebar.header("Select Query Type")
query_type = st.sidebar.radio(
    "Query Type:",
    ["Normal (CSV)", "SQL"],
    index=0 if st.session_state.mode == "normal" else 1
)
st.session_state.mode = "normal" if query_type == "Normal (CSV)" else "sql"

# File uploader
file_type = "csv" if st.session_state.mode == "normal" else "sql"
uploaded_file = st.sidebar.file_uploader(
    f"Choose a {file_type.upper()} file",
    type=file_type,
    key=f"{st.session_state.mode}_uploader"
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

process_btn = st.sidebar.button("Process Data", use_container_width=True)

# Process data
if process_btn and st.session_state.uploaded_file is not None and not st.session_state.processed:
    with st.spinner("Processing..."):
        if st.session_state.mode == "normal":
            st.session_state.df = pd.read_csv(st.session_state.uploaded_file)
            llm = AzureChatOpenAI(
                openai_api_version=AZURE_API_VERSION,
                azure_deployment=AZURE_DEPLOYMENT,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                temperature=0.3,
            )
            
            custom_system_prompt = """You are a helpful AI data analysis assistant. You help users analyze and interpret their data in any European language or English. When given a task:
1. Detect the language of the user's query and respond in the same language.
2. Understand what analysis or visualization is requested.
3. Perform the analysis on the ENTIRE DataFrame provided.
4. For visualizations, generate clear plots with appropriate titles and labels.
5. Explain findings clearly with relevant statistics.
6. Ensure code is secure and handles errors properly.
The user has a DataFrame 'df' containing their data. Use pandas, numpy, matplotlib, and seaborn to provide insights or visualizations."""
            
            st.session_state.agent = create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                system_prompt=custom_system_prompt
            )
            st.session_state.example_questions = generate_example_questions(st.session_state.df)
            st.session_state.processed = True
        else:
            st.session_state.df = pd.DataFrame()
            st.session_state.processed = True

# Main interface
if st.session_state.processed:
    if st.session_state.mode == "normal" and st.session_state.df is not None:
        st.header("Dataset Preview")
        st.dataframe(st.session_state.df)
        
        st.sidebar.header("Example Questions")
        if st.session_state.example_questions:
            for q in st.session_state.example_questions:
                if st.sidebar.button(f"📊 {q['button_text'][:50]}...", help=q['text'], use_container_width=True):
                    process_question(q['text'])
    
    if st.session_state.mode == "sql" and SQL_AVAILABLE:
        st.header("SQL Query Tool")
        sql_query_text = st.text_area(
            "Enter your question in natural language (will be converted to SQL)",
            height=100,
            placeholder="e.g., What is the average FAMOUNT for each SZSPISID?",
            key="sql_input_area"
        )
        
        sql_execute = st.button("Execute Query", use_container_width=True)
        
        if sql_execute and sql_query_text:
            with st.spinner("Generating SQL query..."):
                sql_query = nl_to_sql(sql_query_text, st.session_state.df)
                if sql_query:
                    st.session_state.sql_query = sql_query
                    st.code(sql_query, language="sql")
                    
                    with st.spinner("Executing SQL query..."):
                        result = execute_sql_query(sql_query, st.session_state.df)
                        if result is not None:
                            st.success("Query executed successfully!")
                            st.dataframe(result)
    
    if st.session_state.mode == "normal":
        st.header("Ask a Question or Request a Visualization")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about your data or request a plot (e.g., 'Plot a bar chart of FAMOUNT')"):
            process_question(prompt)
else:
    st.sidebar.info("Select query type and upload a file to start.")
