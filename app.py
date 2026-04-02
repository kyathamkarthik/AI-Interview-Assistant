import streamlit as st
import cv2
import imageio
import pyttsx3
import PyPDF2
import speech_recognition as sr

from speech_to_text import speech_to_text
from answer_analysis import evaluate_answer


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Interview Assistant", layout="centered")


# ---------------- TITLE ----------------
st.title("🎤 AI Interview Assistant")
st.markdown("### 🚀 Analyze interview performance using AI")

# -------------- Add Mic Function ----------------
def listen_from_mic():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except:
        st.error("Could not understand audio")
        return ""


# ---------------- RESUME ANALYZER ----------------
def analyze_resume(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text


def score_resume(text):
    score = 0
    keywords = ["python", "machine learning", "ai", "project", "nlp", "deep learning"]

    for word in keywords:
        if word in text.lower():
            score += 1

    return min(score, 10)


# ---------------- AI VOICE ----------------
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# ---------------- FACE DETECTION ----------------
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    img = cv2.imread(image_path)

    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces)


# ---------------- AI FEEDBACK ----------------
def generate_feedback(score, text):
    if score > 0.75:
        return f"Excellent performance! You answered confidently and clearly. Your response: '{text[:100]}...' shows strong communication skills."
    elif score > 0.5:
        return "Good attempt! Improve structure and clarity."
    else:
        return "Needs improvement. Be more confident and structured."


# ---------------- FILE UPLOAD ----------------
video_file = st.file_uploader("📂 Upload Interview Video", type=["mp4"])
resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])


# ---------------- VIDEO ANALYSIS ----------------
if video_file:

    with open("temp.mp4", "wb") as f:
        f.write(video_file.read())

    st.success("✅ Video uploaded successfully!")
    st.video("temp.mp4")

    if st.button("🚀 Analyze Interview"):

        progress = st.progress(0)

        # Speech
        st.info("🎤 Processing speech...")
        progress.progress(25)

        text = speech_to_text("temp.mp4")

        st.subheader("📝 Extracted Answer")
        st.write(text)

        # NLP
        st.info("🧠 Evaluating answer...")
        progress.progress(60)

        expected = "Tell me about yourself"
        score = evaluate_answer(text, expected)

        # Score
        st.subheader("📊 AI Score")
        st.metric("Performance Score", f"{round(score * 100, 2)} %")

        # Feedback
        feedback = generate_feedback(score, text)

        st.subheader("🧠 AI Feedback")
        st.write(feedback)

        # Voice
        final_message = f"Your score is {round(score * 100)} percent. {feedback}"
        speak_text(final_message)

        # Face Detection
        st.info("👁️ Checking face presence...")
        progress.progress(85)

        faces = 0

        try:
            reader = imageio.get_reader("temp.mp4")

            for i, frame in enumerate(reader):
                imageio.imwrite("frame.jpg", frame)

                count = detect_faces("frame.jpg")

                if count > 0:
                    faces = count
                    break

                if i > 20:
                    break

            st.subheader("👤 Faces Detected")
            st.write(faces)

        except:
            st.warning("Face detection skipped")

        progress.progress(100)
        st.success("🎉 Analysis Completed!")


# ---------------- RESUME ANALYSIS ----------------
if resume_file:

    st.info("📄 Analyzing resume...")

    resume_text = analyze_resume(resume_file)
    resume_score = score_resume(resume_text)

    st.subheader("📊 Resume Score")
    st.write(f"{resume_score} / 10")

    if resume_score > 7:
        st.success("Strong resume! Well done.")
    elif resume_score > 4:
        st.warning("Good resume, but can be improved.")
    else:
        st.error("Resume needs improvement. Add more skills and projects.")



# ---------------- AI CHAT INTERVIEWER ----------------

import random

questions = [
    "Tell me about yourself",
    "What are your strengths?",
    "Explain a machine learning project you worked on",
    "What is overfitting in ML?",
    "Why should we hire you?"
]

st.subheader("🤖 AI Interviewer")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
user_input = st.text_input("Your answer:")

# Ask question
if st.button("🎯 Ask Question"):
    question = random.choice(questions)
    st.session_state.chat_history.append(("AI", question))

# Submit answer
if st.button("✅ Submit Answer"):
    if user_input:
        st.session_state.chat_history.append(("You", user_input))

        # Simple scoring logic
        score = len(user_input.split()) / 20
        score = min(score, 1)

        feedback = generate_feedback(score, user_input)

        response = f"Score: {round(score*100)}%. {feedback}"

        st.session_state.chat_history.append(("AI", response))

# Display chat
for sender, msg in st.session_state.chat_history:
    if sender == "AI":
        st.markdown(f"🤖 **AI:** {msg}")
    else:
        st.markdown(f"🧑 **You:** {msg}")
    

if st.button("🎯 Ask Question", key="ask"):
    question = random.choice(questions)

    st.session_state.current_question = question
    st.session_state.chat_history.append(("AI", question))

    # SPEAK QUESTION 🔊
    speak_text(question)

# ------------ MAKE AI ASK (VOICE) ---------------------
if st.button("🎤 Answer with Mic", key="mic"):

    user_input = listen_from_mic()

    if user_input:
        st.session_state.chat_history.append(("You", user_input))

        # scoring
        score = len(user_input.split()) / 20
        score = min(score, 1)

        feedback = generate_feedback(score, user_input)

        response = f"Score: {round(score*100)}%. {feedback}"

        st.session_state.chat_history.append(("AI", response))

        # SPEAK FEEDBACK 🔊
        speak_text(response)