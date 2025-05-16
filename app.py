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
prompt = st.text_input("🧠 Describe your intern requirement:", placeholder="e.g. Remote Canva intern in Mumbai")

# Dummy prompt vector (match size of your embeddings)
def get_dummy_embedding(text):
    np.random.seed(abs(hash(text)) % 10000)
    return np.random.rand(len(data[0]["embedding"]))

# Match logic
if st.button("🔍 Match Resumes") and prompt:
    st.subheader("Top Matches:")
    query_vector = get_dummy_embedding(prompt)
    matches = []

    for item in data:
        vec = np.array(item["embedding"])
        score = cosine_similarity([query_vector], [vec])[0][0]
        matches.append((item["file_name"], item.get("file_id", ""), item.get("sample", ""), score))

    matches.sort(key=lambda x: x[3], reverse=True)

    for name, file_id, sample, score in matches[:5]:
        st.markdown(f"""
        **📄 {name}**  
        💬 _{sample}_  
        🔢 Match Score: **{round(score * 100, 2)}%**  
        {"[👁 View CV](https://drive.google.com/file/d/" + file_id + "/view)" if file_id else "🚫 No link available"}
        ---
        """)
