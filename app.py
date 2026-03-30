import streamlit as st
from speech_to_text import speech_to_text
from answer_analysis import evaluate_answer

st.title("🎤 AI Interview Assistant")

video_file = st.file_uploader("Upload Interview Video", type=["mp4"])

if video_file:
    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    st.success("Video uploaded successfully!")

    if st.button("Analyze Interview"):

        text = speech_to_text("temp.mp4")
        st.subheader("Extracted Answer")
        st.write(text)

        expected = "Tell me about yourself"
        score = evaluate_answer(text, expected)

        st.subheader("Score")
        st.write(score * 100)