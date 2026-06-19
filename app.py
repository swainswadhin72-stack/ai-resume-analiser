##### Packages Used ######
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
import nltk
nltk.download('stopwords')
from sentence_transformers import SentenceTransformer, util
import torch # Sentence-transformers uses torch backend



# ---------------- STEP 1: ADD IMPORTS ----------------
import re
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()
import os
from dotenv import load_dotenv
from pathlib import Path

# Automatically find the path of the current file (app3.py)
env_path = Path(__file__).parent / ".env"

# Load the .env specifically from that path
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("RAPIDAPI_KEY")

import os
from dotenv import load_dotenv
import pymysql

# 2. Load the .env file
load_dotenv()

# 3. Assign variables FROM the environment (CRITICAL STEP)
# This creates the variables so Python knows what DB_USER is.
DB_HOST = os.getenv("SQL_HOST", "localhost")
DB_USER = os.getenv("SQL_USER", "root")     
DB_PASS = os.getenv("SQL_PASSWORD", "2003") 
API_KEY = os.getenv("RAPIDAPI_KEY")         

# 4. Now you can safely use them
print(f"Attempting connection: User={DB_USER}, Host={DB_HOST}")

# Load BERT model once when the app starts
@st.cache_resource # This keeps the model in memory so it's fast
def load_bert():
    return SentenceTransformer('all-MiniLM-L6-v2')

bert_model = load_bert()

SKILL_MAP = {
    "machine learning": ["ml", "machine learning"],
    "data analysis": ["data analysis", "data analytics"],
    "python": ["python"],
    "sql": ["sql", "mysql", "postgres"],
    "deep learning": ["deep learning", "neural networks"],
    "web development": ["html", "css", "javascript", "react", "node"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "django": ["django"],
    "flask": ["flask"]
}

###### Preprocessing functions ######

# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text
