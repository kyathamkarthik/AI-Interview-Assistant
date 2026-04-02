from speech_to_text import speech_to_text
from answer_analysis import evaluate_answer
import cv2

# Face detection setup
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces)


if __name__ == "__main__":

    # Step 1: Speech to text
    text = speech_to_text("interview.mp4")
    print("\nExtracted Answer:\n", text)

    # Step 2: Compare with expected answer
    expected = "Machine learning is a method where systems learn from data."
    score = evaluate_answer(text, expected)

    print("\nAnswer Score:", score)

    # Step 3: Face detection
    faces = detect_face("face.jpg")

    print("\nFaces Detected:", faces)

    # Final Score (simple logic)
    final_score = score * 100

    print("\nFinal Candidate Score:", final_score)