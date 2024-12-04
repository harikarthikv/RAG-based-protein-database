import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import LlamaCpp

# Page config
st.set_page_config(page_title="Protein Database RAG", layout="wide")
st.title("Protein Database Query System")

# Initialize session state for API token
if 'huggingface_api_token' not in st.session_state:
    st.session_state.huggingface_api_token = None

# API Token input
if not st.session_state.huggingface_api_token:
    with st.form("api_token_form"):
        api_token = st.text_input("Enter HuggingFace API Token:", type="password")
        if st.form_submit_button("Submit"):
            st.session_state.huggingface_api_token = api_token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Only show the main app if API token is set
if st.session_state.huggingface_api_token:
    # Database connection
    @st.cache_resource
    def init_db():
        return SQLDatabase.from_uri('mysql+mysqlconnector://newuser@localhost:3306/protein_db_small')

    # LLM setup
    @st.cache_resource
    
    def init_llms():
        # HuggingFace model
        model_path = r"E:\LLM\models\unsloth\Llama-3.2-3B-Instruct-GGUF\Llama-3.2-3B-Instruct-Q8_0.gguf"
        llm_hf = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_length=256,
            temperature=0.5,
            huggingfacehub_api_token=st.session_state.huggingface_api_token,
        )
        

        #model_path = r"E:\LLM\models\harikarthikv\ibs_GGUF\unsloth.Q8_0.gguf"
        
        llm_local = LlamaCpp(
            model_path=model_path, 
            temperature=0.5,
            max_tokens=1024,
            n_ctx=16384,
            n_gpu_layers=1,
            verbose=True,
        )
        
        return llm_hf, llm_local

    # Initialize components
    db = init_db()
    llm_hf, llm_local = init_llms()

    def get_schema(_):
        return db.get_table_info()

    # Setup chains
    sql_prompt = ChatPromptTemplate.from_template("""
    Based on the table schema below, write sql query that would answer the user's question and don't give any description only the sql query:
    {schema}

    Question:{question}
    SQL Query:
    """)

    response_prompt = ChatPromptTemplate.from_template("""
    You are a precise and knowledgeable protein database expert. 
    Your task is to generate a natural language response that:
    1. Stays within 100 words.
    2. Includes only relevant protein information.
    3. Uses accurate scientific terminology.
    4. Avoids speculation, unnecessary details, or elaboration.
    5. Answers the question directly and succinctly.

    Below are the details for your response:

    Table Schema:
    {schema}

    Original Question:
    {question}

    SQL Query Executed:
    {query}

    Query Result:
    {response}

    Please provide your response:
""")

    # Setup chains
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | sql_prompt
        | llm_hf.bind(stop=["\nSQL Result:"])
        | StrOutputParser()
    )

    def run_query(query):
        return db.run(query)

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_schema,
            response=lambda variables: run_query(variables["query"])
        )
        | response_prompt
        | llm_local
    )

    # Streamlit UI for query input
    st.subheader("Ask a question about proteins")
    user_question = st.text_area("Enter your question:", height=100)

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Processing your question..."):
                try:
                    sql_query = sql_chain.invoke({"question": user_question})
                    
                    with st.expander("View SQL Query"):
                        st.code(sql_query, language="sql")
                    
                    response = full_chain.invoke({"question": user_question})
                    
                    st.success("Response:")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

else:
    st.warning("Please enter your HuggingFace API token to continue.") 