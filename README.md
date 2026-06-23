🚀 AI Resume Analyzer

An intelligent web-based application that analyzes resumes using Machine Learning (ML) and Natural Language Processing (NLP). It provides ATS scores, skill recommendations, and real-time job suggestions to help users improve their resumes and career opportunities.

📌 Features
📄 Resume Parsing (PDF)
🤖 AI-based Resume Analysis
📊 ATS Score Calculation
🧠 Skill Extraction & Recommendations
🎯 Job Role Detection
🔍 Resume vs Job Description Matching
🌐 Real-time Job Search (via API)
📍 Location-based Job Suggestions
📈 Admin Dashboard with Analytics
💬 User Feedback System
🛠️ Tech Stack

Frontend:

Streamlit

Backend:

Python

Machine Learning & NLP:

Scikit-learn
NLTK
Sentence Transformers (BERT)

Libraries Used:

pyresparser
pdfminer3
pandas
plotly
geopy
geocoder

Database:

MySQL

APIs:

JSearch API (RapidAPI)
🧠 How It Works
User uploads resume (PDF)
System extracts text using NLP
Skills and keywords are identified
ATS score is calculated using:
TF-IDF / BERT similarity
Skill matching
Role is predicted (e.g., Data Analyst, Web Developer)
Job recommendations are fetched via API
Suggestions are provided to improve resume
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Setup environment variables (.env)

Create a .env file and add:

SQL_HOST=localhost
SQL_USER=root
SQL_PASSWORD=your_password
RAPIDAPI_KEY=your_api_key
5. Run the application
streamlit run App2.py
📂 Project Structure
├── App2.py
├── Courses.py
├── Uploaded_Resumes/
├── Logo/
├── .env
├── requirements.txt
└── README.md
📊 Database Schema
user_data
Name
Email
Resume Score
Skills
Predicted Field
Timestamp
user_feedback
Name
Email
Feedback Score
Comments
🔐 Security
Environment variables stored in .env
API keys are not hardcoded
Secure database connection using pymysql
📸 Screenshots (Optional)

Add screenshots of your app here

🎯 Future Improvements
Add login/authentication system
Improve AI scoring with deep learning
Add resume builder feature
Deploy on cloud (AWS / Azure)
