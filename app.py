import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PromptIntern", layout="centered")
st.title("🎯 PromptIntern – Smart Resume Matcher")

# Load index
with open("cv_index.json", "r") as f:
    data = json.load(f)

# Input prompt
prompt = st.text_input("🔍 Describe the requirement:", placeholder="e.g. React intern in Mumbai")

# Dummy embedding
def get_dummy_embedding(text):
    np.random.seed(abs(hash(text)) % 10000)
    return np.random.rand(len(data[0]["embedding"]))

# Display
if st.button("Match Resumes") and prompt:
    st.subheader("🔝 Top Matches")
    query_vector = get_dummy_embedding(prompt)
    matches = []

    for item in data:
        vec = np.array(item["embedding"])
        score = cosine_similarity([query_vector], [vec])[0][0]
        matches.append({
            "name": item.get("name", item["file_name"]),
            "location": item.get("location", "-"),
            "email": item.get("email", "-"),
            "sample": item.get("sample", ""),
            "file_id": item.get("file_id", ""),
            "match": round(score * 100, 2)
        })

    matches = sorted(matches, key=lambda x: x["match"], reverse=True)[:5]

    for m in matches:
        st.markdown(f"""
        <div style='border:1px solid #ddd; padding:10px; border-radius:8px; margin-bottom:10px'>
            <b>Name:</b> {m["name"]} <br>
            <b>Location:</b> {m["location"]} <br>
            <b>Email:</b> {m["email"]} <br>
            <b>Match Score:</b> {m["match"]}% <br>
            <a href="https://drive.google.com/file/d/{m["file_id"]}/view" target="_blank">👁 View CV</a>
        </div>
        """, unsafe_allow_html=True)
