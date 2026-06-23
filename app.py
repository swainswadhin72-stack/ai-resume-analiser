###### Packages Used ######
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

# ---------------- STEP 2: ADD FUNCTIONS ----------------

# ---------------- JOB MATCHING ----------------
def calculate_bert_score(resume_text, job_desc):
    # Convert text to embeddings using BERT
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = bert_model.encode(job_desc, convert_to_tensor=True)

    # Calculate Cosine Similarity using BERT utility
    score = util.pytorch_cos_sim(resume_embedding, job_embedding)
    return round(float(score[0][0]) * 100, 2)

def calculate_final_score(resume_text, job_desc, resume_skills, job_skills):
    # 1. Flatten resume_skills if it's a list of lists
    if any(isinstance(i, list) for i in resume_skills):
        resume_skills = [item for sublist in resume_skills for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    # 2. Flatten job_skills if it's a list of lists
    if any(isinstance(i, list) for i in job_skills):
        job_skills = [item for sublist in job_skills for item in (sublist if isinstance(sublist, list) else [sublist])]

    # Get Semantic Score (BERT)
    bert_score = calculate_bert_score(resume_text, job_desc)

    # Get Keyword Score (Exact skill match)
    skill_score = 0
    if job_skills:
        # Now set() will work because the lists are flat
        match_count = len(set(resume_skills) & set(job_skills))
        skill_score = (match_count / len(job_skills)) * 100

    # Hybrid Calculation
    final_score = (0.7 * bert_score) + (0.3 * skill_score)
    return round(final_score, 2)


import re

# Use this for general text similarity (TF-IDF)
def extract_keywords(text):
    return set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))

# Use this for specific Skill Matching and Gap Analysis
def extract_skills_smart(text):
    text = text.lower()
    found_skills = []
    for main_skill, variations in SKILL_MAP.items():
        for v in variations:
            if v in text:
                found_skills.append(main_skill)
                break
    return list(set(found_skills))


def find_missing_skills(resume_text, job_description):
    resume_skills = extract_skills_smart(resume_text)
    job_skills = extract_skills_smart(job_description)

    missing = set(job_skills) - set(resume_skills)
    return list(missing)


def generate_suggestions(resume_text, missing_skills):
    suggestions = []

    if missing_skills:
        suggestions.append(f"Add these skills: {', '.join(missing_skills)}")

    if "project" not in resume_text.lower():
        suggestions.append("Add 2-3 strong projects")

    if "experience" not in resume_text.lower():
        suggestions.append("Add internship or experience")

    if len(resume_text) < 1200:
        suggestions.append("Increase resume content")

    if not suggestions and not missing_skills:
        suggestions.append("Your resume is strong 👍")

    return suggestions


# ---------------- ROLE DETECTION ----------------
def detect_role(resume_text):
    text = resume_text.lower()
    if "machine learning" in text or "data" in text:
        return "Data Analyst"
    elif "html" in text or "css" in text:
        return "Web Developer"
    elif "android" in text:
        return "Android Developer"
    elif "java" in text:
        return "Software Developer"
    return "Software Engineer"





# ---------------- JOB API ----------------
def fetch_jobs_filtered(query, location=None):
    url = "https://jsearch.p.rapidapi.com/search"
    if not API_KEY:
        st.error("API Key is missing! Check your .env file location.")
    headers = {
        "X-RapidAPI-Key": API_KEY, # Ensure your .env key is loaded
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    # If location is provided, append it to the query for better accuracy
    search_query = query
    if location and location.strip() != "":
        search_query = f"{query} in {location}"

    params = {
        "query": search_query,
        "page": "1",
        "num_pages": "1"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database Stuffs ######
print(f"Attempting connection: User={DB_USER}, Host={DB_HOST}")
# Load the .env file
load_dotenv()

# Assign secrets to variables with default values as a safety net
DB_HOST = os.getenv("SQL_HOST", "localhost")
DB_USER = os.getenv("SQL_USER", "root")     
DB_PASS = os.getenv("SQL_PASSWORD", "2003") 
API_KEY = os.getenv("RAPIDAPI_KEY")         

# Database Connection
connection = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASS,
    db='cv'
)

# Define the cursor (Crucial: Add this line if it's missing)
cursor = connection.cursor()



# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (str(sec_token),str(ip_add),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="AI Resume Analyzer - rupam",
   page_icon='./Logo/recommend.png',
   layout="wide",
   initial_sidebar_state="expanded"
)


###### Main function run() ######


def run():
    
    # Simple header
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #5B21B6;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #6D28D9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 AI Resume Analyzer")
    st.markdown("**Analyze resumes using Machine Learning and Natural Language Processing**")
    st.markdown("*by rupam*")
    st.markdown("---")
    
    st.sidebar.header("Navigation")
    activities = ["User", "Feedback", "About", "Admin"]
    choice = st.sidebar.selectbox("Choose:", activities)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built by [Rupam ](https://github.com/)")

    ###### Creating Database and Table ######

    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)

    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)


    ###### CODE FOR CLIENT SIDE (USER) ######

    if choice == 'User':
        act_name = st.text_input('Name*')
        act_mail = st.text_input('Mail*')
        act_mob  = st.text_input('Mobile Number*')
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng
        geolocator = Nominatim(user_agent="http")
        address = {}
        city = address.get('city', '')
        state = address.get('state', '')
        country = address.get('country', '')  

        st.subheader("📄 Upload Your Resume")
        st.write("Upload your resume in PDF format to get instant analysis and recommendations.")
        
        pdf_file = st.file_uploader("Choose your resume PDF", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('🔮 Analyzing your resume with AI... Please wait'):
                time.sleep(4)
        
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                # Extracting resume_text
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis 🤘**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info 👀**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Degree: '+str(resume_data['degree']))                    
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
                except:
                    pass

                ## Predicting Candidate Experience Level 
                cand_level = ''
                if resume_data['no_of_pages'] < 1:                
                    cand_level = "NA"
                    st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
                elif any(x in resume_text.upper() for x in ['INTERNSHIP', 'INTERNSHIPS']):
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif any(x in resume_text.upper() for x in ['EXPERIENCE', 'WORK EXPERIENCE']):
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                else:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at Fresher level!!''',unsafe_allow_html=True)

                st.subheader("**Skills Recommendation 💡**")
                keywords = st_tags(label='### Your Current Skills', text='See our skills recommendation below',value=resume_data['skills'],key = '1  ')

                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
                n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']
                
                recommended_skills = []
                reco_field = ''
                rec_course = ''

                for i in resume_data['skills']:
                    if i.lower() in ds_keyword:
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System',value=recommended_skills,key = '2')
                        rec_course = course_recommender(ds_course)
                        break
                    elif i.lower() in web_keyword:
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System',value=recommended_skills,key = '3')
                        rec_course = course_recommender(web_course)
                        break
                    elif i.lower() in android_keyword:
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System',value=recommended_skills,key = '4')
                        rec_course = course_recommender(android_course)
                        break
                    elif i.lower() in ios_keyword:
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System',value=recommended_skills,key = '5')
                        rec_course = course_recommender(ios_course)
                        break
                    elif i.lower() in uiux_keyword:
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System',value=recommended_skills,key = '6')
                        rec_course = course_recommender(uiux_course)
                        break
                    elif i.lower() in n_any:
                        reco_field = 'NA'
                        st.warning("** Currently our tool only predicts for Data Science, Web, Android, IOS and UI/UX**")
                        recommended_skills = ['No Recommendations']
                        rec_course = "Sorry! Not Available for this Field"
                        break

                st.subheader("**Resume Tips & Ideas 🥂**")
                resume_score = 0
                sections = {
                    'Objective': 6, 'Summary': 6, 'Education': 12, 'School': 12, 'College': 12,
                    'EXPERIENCE': 16, 'Experience': 16, 'INTERNSHIPS': 6, 'Internships': 6,
                    'SKILLS': 7, 'Skills': 7, 'HOBBIES': 4, 'Hobbies': 4, 'INTERESTS': 5,
                    'ACHIEVEMENTS': 13, 'Achievements': 13, 'CERTIFICATIONS': 12, 'PROJECTS': 19
                }
                for section, weight in sections.items():
                    if section in resume_text:
                        resume_score += weight

                st.subheader("**Resume Score 📝**")
                st.markdown("<style>.stProgress > div > div > div > div { background-color: #d73b5c; }</style>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                for percent_complete in range(min(resume_score, 100)):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)
                st.success(f'** Your Resume Writing Score: {resume_score}**')

                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
                insert_data(str(sec_token), str(ip_add), host_name, dev_user, os_name_ver, str(latlong), city, state, country, act_name, act_mail, act_mob, resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)

                st.header("**Bonus Content 💡**")
                st.video(random.choice(resume_videos))
                st.video(random.choice(interview_videos))

                # ---------------- STEP 3: ATS MATCHING ----------------
                st.markdown("---")
                st.subheader("📊 Resume vs Job Description")

                job_desc = st.text_area("Paste Job Description")

                if st.button("Check Match Score"):
                    if job_desc:
                        score = calculate_final_score(resume_text, job_desc, keywords, ds_course) 
# Note: Ensure the variable names 'keywords' and 'ds_course' match your existing code

                        st.metric("ATS Score", f"{score}%")
                        st.progress(int(score))

                        if score < 50:
                            st.error("Low match")
                        elif score < 75:
                            st.warning("Moderate match")
                        else:
                            st.success("Good match")

                        st.subheader("💡 Smart Suggestions")
                        missing = find_missing_skills(resume_text, job_desc)
                        suggestions = generate_suggestions(resume_text, missing)
                        for s in suggestions:
                            st.write("👉", s)

                        st.subheader("🧠 Missing Skills")
                        if missing:
                            for skill in missing:
                                st.error(f"❌ {skill}")
                        else:
                            st.success("No missing skills 🎯")

                        

                # ---------------- STEP 4: SMART JOB SEARCH ----------------
                st.markdown("---")
                st.subheader("💼 Smart Job Recommendations")

                auto_role = detect_role(resume_text)
                st.subheader("💼 Smart Job Recommendations")
                st.write("🔍 Suggested Role (Auto Detected):")
                st.info(auto_role)
                user_role = st.text_input("✏️ Enter Desired Job Role (optional):")
                final_role = user_role if user_role.strip() != "" else auto_role


                job_location = st.text_input("Enter Location (optional)")

                if st.button("Find Jobs"):
                    jobs = fetch_jobs(final_role)

                    if jobs:
                        for job in jobs[:5]:
                            st.markdown(f"### {job.get('job_title')}")
                            st.write(f"Company: {job.get('employer_name')}")
                            st.write(f"Location: {job.get('job_city')}")
                            st.write(f"[Apply Here]({job.get('job_apply_link')})")
                            st.write("---")
                    else:
                        st.warning("No jobs found")

                st.balloons()

            else:
                st.error('Something went wrong..')                

    elif choice == 'Feedback':   
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        with st.form("my_form"):
            feed_name = st.text_input('Name')
            feed_email = st.text_input('Email')
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
            comments = st.text_input('Comments')
            submitted = st.form_submit_button("Submit")
            if submitted:
                insertf_data(feed_name,feed_email,feed_score,comments,timestamp)    
                st.success("Thanks! Your Feedback was recorded.") 
                st.balloons()    

        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        
        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()
        st.subheader("**Past User Rating's**")
        fig = px.pie(values=values, names=labels, title="Chart of User Rating Score", color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig)

    elif choice == 'About':   
        st.header("About AI Resume Analyzer")
        st.write("This tool uses NLP and Machine Learning to analyze resumes.")
        st.markdown("---")
        st.write("**Developer:** Rupam")

    else:
        st.success('Welcome to Admin Side')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')

        if st.button('Login'):
            if ad_user == 'admin' and ad_password == 'admin@resume-analyzer':
                cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), convert(User_level using utf8), city, state, country from user_data''')
                datanalys = cursor.fetchall()
                plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 'User_Level', 'City', 'State', 'Country'])
                st.success(f"Welcome Admin! Total {plot_data.Idt.count()} users analyzed.")                
                
                cursor.execute('''SELECT * from user_data''')
                data = cursor.fetchall()                
                st.header("**User's Data**")
                st.dataframe(pd.DataFrame(data))
            else:
                st.error("Wrong ID & Password Provided")

run()
