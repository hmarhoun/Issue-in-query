import streamlit as st
st.set_page_config(page_title="Talk To My Data", page_icon=":guardsman:", layout="wide")
import pandas as pd
import plotly.express as px
import sqlite3
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import altair as alt
from st_aggrid import AgGrid
from distutils import errors
from distutils.log import error
import streamlit as st
import numpy as np
from itertools import cycle
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import joblib
import io
import sys
import pickle
import cloudpickle
#import pages.premium as premium
from streamlit_extras.switch_page_button import switch_page
import json
import fcntl
import tempfile
import csv
import re
import uuid
from urllib import parse


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CACHE_DIR = "cache"
CACHE_FILE = "df.pkl"

class SessionManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.sessions = {}

    def session_exists(self, session_id):
        session_dir = self.get_session_dir(session_id)
        return os.path.exists(session_dir)

    def get_session_dir(self, session_id):
        """Returns the directory path for the given session ID."""
        return os.path.join(self.cache_dir, session_id)

    def create_session(self, session_id):
        """Creates a new session directory for the given session ID."""
        session_dir = self.get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)

    def get_cache_file(self, session_id):
        """Returns the cache file path for the given session ID."""
        return os.path.join(self.get_session_dir(session_id), "cache.pkl")

    def get_sqlite_file(self, session_id, table_name):
        """Returns the SQLite database file path for the given session ID and table name."""
        return os.path.join(self.get_session_dir(session_id), f"{table_name}.db")

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        return self.sessions[session_id]

    def set_session(self, session_id, data):
        self.sessions[session_id] = data

    def store_df_in_cache(self, session_id, df):
        cache_file = os.path.join(self.get_session_dir(session_id), CACHE_FILE)
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)

    def get_df_from_cache(self, session_id):
        cache_file = os.path.join(self.get_session_dir(session_id), CACHE_FILE)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            return None


    def store_df_in_session(self, session_id, df):
        with open(os.path.join(self.get_session_dir(session_id), "df.pkl"), "wb") as f:
            pickle.dump(df, f)

    def get_df_from_session(self, session_id):
        with open(os.path.join(self.get_session_dir(session_id), "df.pkl"), "rb") as f:
            return pickle.load(f)

session_manager = SessionManager(CACHE_DIR)


# get the session ID from the URL or prompt the user to enter it
url = st.experimental_get_query_params()
if 'session_id' in url:
    session_id = url['session_id']
else:
    session_id = ''

# prompt the user to enter a valid session ID
session_input_key = "session_input"
session_id = st.text_input("Enter your session ID", key=session_input_key)

# create a new session if it doesn't exist
if not session_manager.session_exists(session_id):
    st.info(f"New session created. Your session ID is {session_id}.")
    session_manager.create_session(session_id)
    url = st.experimental_get_query_params()
    url['session_id'] = session_id
    st.experimental_set_query_params(**url)


def convert_csv_to_sqlite(csv_file_path, table_name, session_id):
    # Get the column names from the CSV file
    with open(csv_file_path, "r") as csv_file:
        dialect = csv.Sniffer().sniff(csv_file.read(1024))
        csv_file.seek(0)
        csv_reader = csv.reader(csv_file, dialect)
        headers = next(csv_reader)

    # modify the column headers to remove special characters and spaces
    headers = [re.sub(r'[^\w\s]', '', h).replace(' ', '_') for h in headers]
    #print (headers)
    # Connect to the SQLite database file
    sqlite_file_path = session_manager.get_sqlite_file(session_id, table_name)
    conn = sqlite3.connect(sqlite_file_path)
    cursor = conn.cursor()

    # Create the table
    column_definitions = []
    for header in headers:
        column_definitions.append(f"{header} TEXT NOT NULL")
    table_create_query = f'''CREATE TABLE IF NOT EXISTS "{table_name}" ({','.join(column_definitions)});'''
    #print(table_create_query)
    try:
        cursor.execute(table_create_query)
    except Exception as e:
        st.error("The file provided includes headers names with special charcters !")

    # Read the CSV file and insert the data into the table
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, dialect)
        next(csv_reader)
        for row in csv_reader:
            values = ",".join([f"'{value}'" for value in row])
            insert_query = f"INSERT INTO {table_name} ({','.join(headers)}) VALUES ({values});"
            cursor.execute(insert_query)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    return sqlite_file_path

def generate_promptini(sqlite_file_path, table_name=None):
    conn = sqlite3.connect(sqlite_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if len(tables) == 0:
        st.error(f"No tables found in {sqlite_file_path}.")
        return None
    if table_name is None:
        table_name = tables[0][0]
    else:
        table_names = [t[0] for t in tables]
        if table_name not in table_names:
            st.error(f"Table {table_name} not found in {sqlite_file_path}. Available tables: {', '.join(table_names)}.")
            return None
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    headers = [description[0] for description in cursor.description]
    promptini = f"""### SQLite table, with its properties:
        # 
        # {table_name} ({", ".join(headers)})
        #
        ### """
    conn.close()
    return promptini


api_key_file = "api_key.json"
validated = False

def test_api_key():
    try:
        api_call_made = joblib.load("api_call_made.joblib")
    except:
        api_call_made = False

    if not api_call_made:
        with open("api_call_made.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                completions = openai.Completion.create(
                    model="text-ada-001",
                    prompt="write affirmative ",
                    temperature=0,
                    max_tokens=40,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop= ["."]
                )
                valid_response = completions.choices[0].text.lstrip()
                joblib.dump(True, "api_call_made.joblib")
                #print (valid_response)
                return valid_response
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    else:
        return None

if not os.path.exists(api_key_file):
    api_key = None
    if not validated:
        api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", key="unique_api_key")
    if api_key:
        Display = True 
        openai.api_key = api_key
        session_dir = session_manager.get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        api_key_file = os.path.join(session_dir, "api_key.json")
        with open(api_key_file, "w") as f:
            api_key_dict = {"api_key": api_key}
            f.write(json.dumps(api_key_dict))
else:
    try:
        session_dir = session_manager.get_session_dir(session_id)
        api_key_file = os.path.join(session_dir, "api_key.json")
        with open(api_key_file, "r") as f:
            api_key = json.load(f)["api_key"]
            openai.api_key = api_key
    except FileNotFoundError:
        st.sidebar.error("Failed to read the API Key from the file.")
    except Exception as e:
        st.sidebar.error("An error occurred while reading the API Key from the file.")

 



def translate_to_sql(session_id, promptini, text_input, table_name):
    completions = openai.Completion.create(
        model="code-davinci-002",
        prompt=promptini + text_input + "\n SELECT",
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )
    sql_query = 'SELECT ' + completions.choices[0].text.lstrip()
    st.success(sql_query)
    sqlite_file_path = session_manager.get_sqlite_file(session_id, table_name)
    conn = sqlite3.connect(sqlite_file_path)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

# allow the user to upload a file and convert it to SQLite
csv_file_path = st.file_uploader("Upload your file", type=["csv"])
if csv_file_path is not None:
    file_extension = csv_file_path.name.split(".")[-1]
    if file_extension == 'csv':
        table_name = os.path.splitext(os.path.basename(csv_file_path.name))[0]
        #print (table_name)
        with open(os.path.join(session_manager.get_session_dir(session_id), f"{table_name}.csv"), "wb") as f:
            f.write(csv_file_path.read())

        # now you can use the file for processing
        csv_file_path = os.path.join(session_manager.get_session_dir(session_id), f"{table_name}.csv")
        sqlite_file_path = convert_csv_to_sqlite(csv_file_path, table_name, session_id)
        promptini = generate_promptini(sqlite_file_path, table_name)
        


st.title("Talk To My Data")
text_input = st.text_input("Enter your text")
if st.button("Translate to SQL"):
    try:
        df = translate_to_sql(session_id, promptini, text_input, table_name)
    except Exception as e:
        st.error("Sorry, we encountered an error while processing your request.")
    try:
        session_manager.store_df_in_cache(session_id, df)
    except Exception as e:
        st.error("Sorry, You need to Provide your OpenAI API Key.")
else:
    try:
        df = session_manager.get_df_from_cache(session_id)
    except Exception as e:
        st.error("Sorry, we encountered an error while processing your request.")
if df is not None:
    #Example controlers
    #st.sidebar.header("Filtering and Display options")
    #sample_size = st.sidebar.number_input("rows", min_value=4, value=50)
    grid_height = 300
    #enterprise modules
    enable_enterprise_modules = st.sidebar.checkbox("Enable Enterprise Modules")
    if enable_enterprise_modules:
        enable_sidebar =st.sidebar.checkbox("Enable grid sidebar", value=False)
    else:
        enable_sidebar = False
    st.sidebar.header("Table display options")
    #features
    fit_columns_on_grid_load = st.sidebar.checkbox("Fit Grid Columns on Load", value=False)
    enable_pagination = st.sidebar.checkbox("Enable pagination", value=False)
    if enable_pagination:
        st.sidebar.subheader("Pagination options")
        paginationAutoSize = st.sidebar.checkbox("Auto pagination size", value=True)
        if not paginationAutoSize:
            paginationPageSize = st.sidebar.number_input("Page size", value=5, min_value=0, max_value=20)
        st.sidebar.text("___")
    #st.write(df)
    #Infer basic colDefs from dataframe types
    gb = GridOptionsBuilder.from_dataframe(df)
    #customize gridOptions
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
    #configures last row to use custom styles based on cell's value, injecting JsCode on components front end
    if enable_sidebar:
        gb.configure_side_bar()
    if enable_pagination:
        if paginationAutoSize:
            gb.configure_pagination(paginationAutoPageSize=True)
        else:
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    #Display the grid
    #st.header("Display My Data")
    #st.write(df)
    #grid_response = AgGrid(
    #    df, 
    #    gridOptions=gridOptions,
    #    height=grid_height, 
    #    width='100%',
    #    fit_columns_on_grid_load=fit_columns_on_grid_load,
    #    allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
    #    enable_enterprise_modules=enable_enterprise_modules
    #    )
    #df = grid_response['data']
    #res = grid_response['data']
    st.header("Play With My Data")
    def filter_dataframe(df):
            # Get the columns with unique values less than 7
            columns_to_filter = [
                column for column in df.columns
                #if len(df[column].unique()) < 15
            ]
            #print (columns_to_filter)
            # Create a multiselect for the user to choose columns
            selected_columns = st.sidebar.multiselect(
                "Select columns to filter by",
                options=columns_to_filter,
                default=[]
            )
            # Create a filter widget for each selected column
            for column in selected_columns:
                unique_values = df[column].unique()
                selected_values = st.sidebar.multiselect(
                    f"Select values to include for column '{column}'",
                    options=unique_values,
                    default=[]
                )
                before_filter = df.shape[0]
                df = df[df[column].isin(selected_values)]
                after_filter = df.shape[0]
                st.sidebar.success(f"Number of rows after filtering by {column}: {after_filter} (reduced from {before_filter})")
            return df
    st.sidebar.title("Filter DataFrame")
    filtered_df = filter_dataframe(df)
    #st.write("Filtered DataFrame")
    #st.write(filtered_df)
    grid_response_filtered = AgGrid(
    filtered_df, 
    gridOptions=gridOptions,
    height=grid_height, 
    width='100%',
    fit_columns_on_grid_load=fit_columns_on_grid_load,
    enable_enterprise_modules=enable_enterprise_modules
    )
    filtered_df = grid_response_filtered['data']
    tab1, tab2= st.tabs(["Visualize Data myself", "Visualize My Data using AI"])
    with tab1:
        st.header("Visualize Data myself")
        if st.checkbox("Visualize Data myself"):
            chart_type = st.selectbox("Select the chart type", options=["bar", "line", "Pie"])
            if chart_type: # Only show the next select boxes if chart_type is selected
                column1 = st.selectbox("Select the column for x-axis", options=df.columns)
                column2 = st.selectbox("Select the column for y-axis", options=df.columns)
                if chart_type == "bar":
                    fig = px.bar(filtered_df, x=column1, y=column2, color_discrete_sequence = ['#F63366'])
                    st.plotly_chart(fig)
                elif chart_type == "line":
                    fig = px.line(df, x=column1, y=column2, color_discrete_sequence = ['#F63366'])
                    st.plotly_chart(fig)
                elif chart_type == "Pie":
                    fig = px.pie(df, values=column2, names=column1)
                    st.plotly_chart(fig)
    with tab2:
        def get_cached_api_response(session_id):
            try:
                with open(session_manager.get_cache_file(session_id), "rb") as f:
                    response = cloudpickle.load(f)
            except FileNotFoundError:
                response = None
            return response

        def store_api_response_in_cache(session_id, response):
            with open(session_manager.get_cache_file(session_id), "wb") as f:
                cloudpickle.dump(response, f)

        st.header("Visualize My Data using AI")
        header = '|'.join(df.columns)
        output = "table = df.head(4).astype(str).apply(lambda x: '|'.join(x), axis=1)\n"
        output += f"print('{header}')\n"
        output += "for i, row in table.iteritems():\n"
        output += "    print(f'{i + 1}|{row}')"
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        exec(output)
        sys.stdout = sys.__stdout__
        data_query = capturedOutput.getvalue()
        col_info = [f"{col} ({df[col].dtype})" for col in df.columns]

        def get_col_types(df):
            """Return the data types of the first row in each column."""
            pattern_int = re.compile(r'^[-+]?\d+$')
            pattern_float = re.compile(r'^[-+]?\d*\.\d+$')
            col_types = {}
            for col in df.columns:
                first_value = str(df.loc[0, col])
                if pattern_int.match(first_value):
                    col_types[col] = int
                elif pattern_float.match(first_value):
                    col_types[col] = float
                else:
                    col_types[col] = str
            return pd.Series(col_types)
        col_types = get_col_types(df)
        text_context = f"""Please generate a visualization of the data using the Python plotting library Plotly and the Streamlit framework.
        The data is stored in the following pandas DataFrame: {col_types}.  
        Write only a single streamlit code that is once executed will display the graph. End systematically the code with a semicolumn ";" \n The graph could be a scatter plots, area charts, bar charts, box plots, histograms, heatmaps, subplots, multiple-axes, polar charts, and bubble charts...
        Example of response expected: st.line_chart(df.groupby('Date')['TransactionID'].count());
        Make sure to analyze the values of each row to avoid unsupported operand type(s) for /: 'str' and 'int' \n
        """
        text = "Here the dataframe with some values: \n "
        user_input = st.text_input("What do you want to display:")
        prompt_api = text_context + user_input + "\n" + text + "\n"  + data_query 
        #st.info(prompt_api)
        @st.cache(suppress_st_warning=True, allow_output_mutation=True)
        def get_api_response(session_id, prompt):
            completions = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt + "\n st.",
                temperature=0,
                max_tokens=800,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["#", ";"]
            )
            response_plot = completions.choices[0].text.lstrip()
            store_api_response_in_cache(session_id, response_plot)
            return response_plot

        if st.button("Compute Visualization"):
            session_data = session_manager.get_session(session_id)
            prev_input = session_data.get('prev_input')
            response_plot = get_cached_api_response(session_id)
            if prev_input != user_input or response_plot is None:
                session_data['prev_input'] = user_input
                prompt_api = text_context + user_input + "\n" + data_query
                response_plot = get_api_response(session_id, prompt_api)
            else:
                st.success("API response loaded from cache.")
            st.success(f"plot query:{response_plot}")
            st.write(response_plot)
            a = "st." + response_plot
            #b = st.line_chart(df.groupby('Date')['TransactionID'].count())
            exec(a)




