import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load index
with open("cv_index.json", "r") as f:
    data = json.load(f)

# Page setup
st.set_page_config(page_title="PromptIntern", layout="centered")
st.title("🎯 PromptIntern – AI Resume Finder")

# Prompt input
prompt = st.text_input("Enter a prompt:", placeholder="Looking for React intern in Mumbai who knows Canva")

# Dummy embedding for now
def get_dummy_embedding(text):
    # Fake embedding to match with dummy CVs
    np.random.seed(abs(hash(text)) % 10000)
    return np.random.rand(384)

# Match logic
if st.button("🔍 Match Resumes") and prompt:
    st.subheader("Top Matches:")
    query_vector = get_dummy_embedding(prompt)
    matches = []
    for item in data:
        vec = np.array(item["embedding"])
        score = cosine_similarity([query_vector], [vec])[0][0]
        matches.append((item["file_name"], score))

    matches.sort(key=lambda x: x[1], reverse=True)
    for name, score in matches[:5]:
        st.write(f"📄 **{name}** — {round(score*100, 2)}% match")
