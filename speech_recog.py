import speech_recognition as sr
import pyaudio
import time

# Method definition for converting raw speech into text
#       recognizer: A `Recognizer` object from the `speech_recognition` library.
#       microphone: Input source for the recognition
def convert_voice_to_text(recognizer, microphone):
    
    with microphone as source:
        print("Listening...")
        # reduces noise from the background
        recognizer.adjust_for_ambient_noise(source, duration=2)
        audio = recognizer.listen(source)
        try:
            #speech recogniser starts to listen to source and prints result
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print("Could not request results") 

# It splits the speech text to perform commands
def listen_for_command(command):
    try:
        # ensures text no longer than 10 words for testing
        words = command.split()
        if len(words) < 10:
         if words[0] == "exit":
            return "exit"
         
         if words[0] == "face" and words[1] == "detection":
            print("Face detection is loading...")
            return "face_detection"
         
         if words[0] == "train" and words[1] == "model":
            print("Model training in progress...")
            time.sleep(2)
            return "train_model"
         
         if words[0] == "run" and words[1] == "model":
            print("Model is starting to run...")
            time.sleep(2)
            return "run_model"
         
    except IndexError:
        return "Invalid input format"