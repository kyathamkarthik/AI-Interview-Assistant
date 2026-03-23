import whisper

def speech_to_text(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["text"]

if __name__ == "__main__":
    text = speech_to_text("interview.mp4")
    print("\nExtracted Text:\n")
    print(text)