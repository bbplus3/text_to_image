import numpy as np
import pandas as pd
import warnings
import streamlit as st
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import torch

warnings.filterwarnings('ignore')

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("t_bbe.csv")
    book_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi', 40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter', 62: '1 John',
        63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }
    df['Book Name'] = df['b'].map(book_names)
    return df, book_names

data, book_numbers = load_data()

# Load Stable Diffusion pipeline
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_pipeline()

# Streamlit App
st.title("🖼️ Bible Passage Text-to-Image Generator")

with st.form("user_input"):
    col1, col2 = st.columns(2)
    with col1:
        input_book = st.selectbox("Select Book", book_numbers.values())
        input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
    col3a, col3b = st.columns(2)
    with col3a:
        start_verse = st.number_input("Start Verse", min_value=1, max_value=176, value=1, step=1)
    with col3b:
        end_verse = st.number_input("End Verse", min_value=1, max_value=176, value=1, step=1)

    col4a, col4b = st.columns(2)
    with col4a:
        style = st.selectbox("Select Style", ["realistic", "oil painting", "digital art", "sketch", "fantasy art"])
    with col4b:
        resolution = st.selectbox("Image Resolution", ["512x512", "768x768"])  # Safe resolutions

    submitted = st.form_submit_button("Generate Image")

if submitted:
    selected_verses = data.loc[
        (data['Book Name'] == input_book) &
        (data['c'].astype(str) == str(input_chapter)) &
        (data['v'].astype(int) >= start_verse) &
        (data['v'].astype(int) <= end_verse)
    ]

    if not selected_verses.empty:
        passage = ' '.join(selected_verses['t'].tolist())
        truncated_passage = passage[:300]  # Limit length for safety
        prompt = f"A child with dark hair, olive skin, and green eyes. {truncated_passage}, style: {style}"

        width, height = map(int, resolution.split("x"))

        st.write(f"**Input Passage:** {passage}")
        st.write("### 🖼️ Generated Image:")
        image = pipe(prompt, height=height, width=width).images[0]
        st.image(image, width=600)
    else:
        st.write("Passage not found.")
