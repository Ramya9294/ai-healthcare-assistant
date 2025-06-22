# AI-Powered Healthcare Assistant (End-to-End)
# Tools: Kaggle Dataset + LangChain Agent + ML + Orange Tool + Streamlit

# ------------------------------
# 0. Required Installations
# ------------------------------
# Make sure to install the required packages:
# pip install pandas scikit-learn joblib streamlit langchain transformers pypdf matplotlib

# ------------------------------
# 1. Data Handling & Preprocessing
# ------------------------------

with open("requirements.txt", "w") as f:
    f.write("""pandas
scikit-learn
joblib
streamlit
langchain
transformers
pypdf
sentence-transformers
faiss-cpu
matplotlib""")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json

# Load dataset (downloaded from Kaggle)
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "diabetes_model.pkl")

# ------------------------------
# 2. Orange ML (Optional GUI for data analysis)
# - Load same CSV file in Orange
# - Use widgets: File → Select Columns → Test & Score → Random Forest
# - Export predictions if needed

# ------------------------------
# 3. RAG using LangChain Agent (Local PDF Q&A)
# ------------------------------
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline

# Load PDF
loader = PyPDFLoader("diabetes_guide.pdf")
docs = loader.load()

# Embedding and Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Setup retriever QA tool
local_pipeline = pipeline("text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=local_pipeline)
retriever = vectorstore.as_retriever()
qa_chain = load_qa_chain(llm, chain_type="stuff")

def rag_tool_func(q):
    rel_docs = retriever.get_relevant_documents(q)
    return qa_chain.run(input_documents=rel_docs, question=q)

rag_tool = Tool(
    name="MedicalPDF_QA",
    func=rag_tool_func,
    description="Answers health questions from a diabetes medical guide PDF"
)

agent = initialize_agent([rag_tool], llm, agent="zero-shot-react-description", verbose=False)

# ------------------------------
# 4. Streamlit Frontend
# ------------------------------
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title("🩺 AI Healthcare Assistant")
st.write("Predict disease risk + Ask health questions using offline tools")

uploaded = []  # ✅ Prevents NameError when 'uploaded' is referenced later


# Input fields
preg = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 1)

user_question = st.text_input("Ask a health question")

# Predict button
if st.button("🔎 Predict Risk"):
    user_data = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, diabetes_pedigree, age]],
                              columns=X.columns)
    model = joblib.load("diabetes_model.pkl")
    prediction = model.predict(user_data)[0]
    risk = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"
    st.subheader("🧪 Prediction Result")
    st.success(risk)

    # Save to offline cache (simulate local device)
    local_dir = "offline_data"
    os.makedirs(local_dir, exist_ok=True)
    record = user_data.copy()
    record["Risk"] = risk
    record["Timestamp"] = datetime.now().isoformat()
    record.to_json(f"{local_dir}/entry_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", orient="records")

# Q&A
if user_question:
    st.subheader("📖 Health Info:")
    with st.spinner("Looking in medical documents..."):
        response = agent.run(user_question)
    st.info(response)



# Glucose Time Series (Simulated CGM-style)
st.subheader("📈 Real-Time Glucose Trend (Simulated)")
time_series = [datetime.now() - timedelta(minutes=15*i) for i in range(12)][::-1]
glucose_series = [130, 128, 125, 126, 160, 170, 168, 165, 170, 180, 200, 305]  # Simulated values

fig2, ax2 = plt.subplots()
ax2.plot(time_series, glucose_series, marker='o', color='cyan', linewidth=2)
ax2.set_ylabel("Glucose (mg/dL)")
ax2.set_xlabel("Time")
ax2.set_title("Recent Glucose Levels")
ax2.axhline(180, color='orange', linestyle='--', label="Caution Threshold")
ax2.axhline(300, color='red', linestyle='--', label="High Alert")
ax2.legend()
fig2.autofmt_xdate()
st.pyplot(fig2)

# Glucose Histogram
st.subheader("📊 Glucose Level Distribution")
fig, ax = plt.subplots()
ax.hist(data["Glucose"], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("Glucose Level")
ax.set_ylabel("Count")
ax.set_title("Distribution of Glucose Levels in Dataset")
st.pyplot(fig)


# ------------------------------
# 6. Simulated Offline-to-Cloud Sync Button
# ------------------------------

st.subheader("☁️ Sync Offline Records to Cloud (Simulated)")
if st.button("Upload Cached Data"):
    uploaded = []
    local_dir = "offline_data"
    if os.path.exists(local_dir):
        for filename in os.listdir(local_dir):
            if filename.endswith(".json"):
                with open(os.path.join(local_dir, filename), "r") as f:
                    record = json.load(f)
                    uploaded.append(record)
                os.remove(os.path.join(local_dir, filename))  # simulate upload/delete

    if uploaded:
        st.success(f"✅ Uploaded {len(uploaded)} record(s) to cloud server (simulated).")
        for rec in uploaded:
            st.json(rec)
    else:
        st.info("📭 No cached records to sync.")

# 7. Doctor Dashboard (Simulated)
st.subheader("🩻 Doctor Dashboard")
dashboard_dir = "cloud_data"
if not os.path.exists(dashboard_dir):
    os.makedirs(dashboard_dir)

uploaded = uploaded if "uploaded" in locals() else []  # Ensure uploaded is defined

# Simulate storing uploaded records for doctor
for record in uploaded:
    timestamp = record[0]["Timestamp"].replace(":", "-").replace("T", "_")
    with open(f"{dashboard_dir}/record_{timestamp}.json", "w") as f:
        json.dump(record, f)

# Display all uploaded patient summaries
records = []
for file in os.listdir(dashboard_dir):
    if file.endswith(".json"):
        with open(os.path.join(dashboard_dir, file), "r") as f:
            records.extend(json.load(f))

if records:
    df = pd.DataFrame(records)
    st.dataframe(df)
    st.bar_chart(df["Glucose"])
else:
    st.info("📁 No patient data available yet. Upload and sync records first.")
