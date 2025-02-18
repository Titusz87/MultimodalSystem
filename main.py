from sklearn.metrics import classification_report, confusion_matrix
from speech_recog import convert_voice_to_text, listen_for_command
from sklearn.model_selection import train_test_split
from knn_train_test import train_knn, test_knn
from load_data import start_load_data
import speech_recognition as sr
import numpy as np
import face_live
import pickle

#   AAI 202 
#   Assessment 3 project
#
#   Tibor Titusz Tarcsai

# Main program consisting of a while loop to perform on speech commands.
# 4 commands including face_detection, train_model, run_model and exit that halts the program.
# train_model and run_model without training data/images wont run.

def main():
    #Initialisation of the microphone and the regonizer object for speech recognition
    recognizer = sr.Recognizer()    
    microphone = sr.Microphone()
    command = None
    model = None
   
    while True:
         captured_text = convert_voice_to_text(recognizer, microphone)
         command = listen_for_command(captured_text)
         if command == "exit":
             return
         if command == "face_detection":
             face_live.start_face_detection(0, None, True)
         if command == "train_model":
             break
         if command == "run_model":
             break
         
    if(command == "train_model"):

        # Load data from "labels.txt"
        images, labels = start_load_data("labels.txt")
        images_array = np.array(images)

        # Converts each picture into 1D vectors
        flattened = [image.flatten() for image in images_array] 
        flattened_array = np.array(flattened)
   
        # Splits data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(flattened_array, labels, test_size=0.2)

        # Trains kNN model
        knn = train_knn(X_train, y_train)

        # Saves the trained model into a file
        with open('knn_model.pkl', 'wb') as model_file:
            pickle.dump(knn, model_file)
           
        # Tests the kNN model
        predictions = test_knn(knn, X_test, y_test)

        # Outputs the Classification report and Confusion matrix
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

    if(command == "run_model"):
         # It loads the saved trained Knn model from file
         with open('knn_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
         face_live.start_face_detection(0, model,  True)
    
    #Ensures to start the loop again for incomming speech commands 
    main()
    
if __name__ == "__main__":
    main()