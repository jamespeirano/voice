import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import requests
import keyboard
import tempfile
import wave
import time
import pyautogui
import pyperclip
import google.genai as genai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("VOICE_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


def save_wav(audio, fs):
    # Save audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        with wave.open(tmpfile, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        return tmpfile.name

def transcribe(audio, fs):
    wav_path = save_wav(audio, fs)
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": MODEL,
        "language": "en"
    }
    try:
        with open(wav_path, "rb") as f:
            files = {
                "file": (os.path.basename(wav_path), f, "audio/wav"),
            }
            response = requests.post(url, headers=headers, files=files, data=data)
        # File is now closed, safe to remove
        for _ in range(5):
            try:
                os.remove(wav_path)
                break
            except PermissionError:
                time.sleep(0.1)
        if response.ok:
            return response.json().get("text", "")
        else:
            print("Error:", response.text)
            return ""
    except Exception as e:
        print(f"Exception during transcription: {e}")
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        return ""

def get_highlighted_text():
    old_clipboard = pyperclip.paste()
    copied_text = ""
    try:
        pyperclip.copy("")
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.3)
        copied_text = pyperclip.paste()
    finally:
        # Always restore the original clipboard
        pyperclip.copy(old_clipboard)
    return copied_text if copied_text else ""

def gemini_ask(prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt]
    )
    return response.text if hasattr(response, 'text') else str(response)


def main():
    print("Hold F8 to record, release to transcribe and type at cursor...")
    print("Hold F7 to record and send highlighted text + voice to Gemini...")
    print("Hold f23 to do the same as F7, but with a custom prompt from custom_prompt1.txt...")
    fs = 16000
    while True:
        if keyboard.is_pressed('f8'):
            keyboard.wait('f8')
            print("Listening... (release F8 to stop)")
            audio = []
            recording = True

            def callback(indata, frames, time_info, status):
                if recording:
                    audio.append(indata.copy())

            with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
                while keyboard.is_pressed('f8'):
                    sd.sleep(50)
                recording = False
            print("Transcribing...")
            if audio:
                audio_np = np.concatenate(audio, axis=0)
                text = transcribe(audio_np, fs)
                print("You said:", text)
                if text.strip():
                    print("Typing at cursor...")
                    pyautogui.typewrite(text)
            else:
                print("No audio captured.")
        elif keyboard.is_pressed('f7'):
            keyboard.wait('f7')
            print("Listening for Gemini... (release F7 to stop)")
            audio = []
            recording = True

            def callback(indata, frames, time_info, status):
                if recording:
                    audio.append(indata.copy())

            with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
                while keyboard.is_pressed('f7'):
                    sd.sleep(50)
                recording = False
            print("Transcribing audio for Gemini...")
            highlighted_text = get_highlighted_text()
            if audio:
                audio_np = np.concatenate(audio, axis=0)
                voice_text = transcribe(audio_np, fs)
                print("Highlighted text:", highlighted_text)
                print("Voice said:", voice_text)
                prompt = f"Context: {highlighted_text}\n\nUser says: {voice_text}"
                print("Sending to Gemini...")
                try:
                    gemini_response = gemini_ask(prompt)
                    print("Gemini response:", gemini_response) 
                except Exception as e:
                    print("Error from Gemini:", e)
            else:
                print("No audio captured for Gemini.")
        elif keyboard.is_pressed('f23'):
            keyboard.wait('f23')
            print("Listening for Gemini with custom prompt... (release f23 to stop)")
            audio = []
            recording = True

            def callback(indata, frames, time_info, status):
                if recording:
                    audio.append(indata.copy())

            with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
                while keyboard.is_pressed('f23'):
                    sd.sleep(50)
                recording = False
            print("Transcribing audio for Gemini...")
            highlighted_text = get_highlighted_text()
            custom_prompt = ""
            try:
                with open("custom_prompt1.txt", "r", encoding="utf-8") as f:
                    custom_prompt = f.read().strip()
            except Exception as e:
                print("Could not read custom_prompt1.txt:", e)
            if audio:
                audio_np = np.concatenate(audio, axis=0)
                voice_text = transcribe(audio_np, fs)
                print("Highlighted text:", highlighted_text)
                print("Voice said:", voice_text)
                prompt = f"{custom_prompt}\n\nContext: {highlighted_text}\n\nUser says: {voice_text}"
                print("Sending to Gemini with custom prompt...")
                try:
                    gemini_response = gemini_ask(prompt)
                    print("Gemini response:", gemini_response)
                except Exception as e:
                    print("Error from Gemini:", e)
            else:
                print("No audio captured for Gemini.")
        else:
            time.sleep(0.05)

if __name__ == "__main__":
    main() 