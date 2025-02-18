import os
import cv2

def start_face_detection(pictureNumber, model, isRunning = False):
    if(isRunning == True):
         
         #Reads the cascade model from location and saves it as a variable
         cascPath = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
         faceCascade = cv2.CascadeClassifier(cascPath)

        #Initialises the videocamera to read on for input
         video_capture = cv2.VideoCapture(0)
         pictureNumber = pictureNumber
         picture = None
         faces = None
         while True:
                 # Captures frame-by-frame
                 ret, frame = video_capture.read()    
                 if not ret:
                    print("Error: Could not read frame")
                    break
                 
                 # Displays the resulting frame
                 cv2.imshow('Video', frame)    
                 key = cv2.waitKey(1) & 0xFF
                 if key == ord('c'):  # Captures image on 'c' keypress
                        cv2.imwrite(f'image_{pictureNumber}.png', frame)
                        print("Image captured!")
                        picture_name = f'image_{pictureNumber}.png'
                        picture = cv2.imread(picture_name)

                        cv2.destroyAllWindows()
                        break
                 elif key == ord('q'):  # Quits on 'q' keypress
                         break         
         
         video_capture.release()

        #Face processing if no face recognition is involved
         if(model != None):     
                face_processing(picture,faces, faceCascade, pictureNumber, model, True)
         else:
            #Face processing if face recognition is involved
            face_processing(picture,faces, faceCascade, pictureNumber, None , False)
            pictureNumber += 1

            # When everything is done, release the capture  
            cv2.destroyAllWindows()

            #Saves pictures name with location and their labels
            with open("labels.txt", "r") as f:
                line_count = sum(1 for line in f)
                if line_count < 31:
                    start_face_detection(pictureNumber,True)
    else:
         return

def face_processing(picture, face_list, faceCascade, pictureNumber, model, Isrecognition_ON = False):
      if(picture.any()):

        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) #image grayscaling to reduce color channel complexity

        # Detects faces in the image
        face_list = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            )
        print(f"Found {len(face_list)} faces!")
        
      if(Isrecognition_ON == True):
            
            # Image preprocessing stage. Converts the picture into 1D vectors 
            # and reshapes them so Knn model can accept them as input.
            img_name = f'image_{pictureNumber}.png'
            img = cv2.imread(img_name)
            img_flattened = img.flatten()
            img_reshaped = img_flattened.reshape(1, -1)
            
            # The first class prediction is saved into a variable
            prediction = model.predict(img_reshaped)
            label = prediction[0]

            #Draws a rectangle around the face and add the label
            for (x, y, w, h) in face_list:
                cv2.rectangle(picture, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(picture, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Faces found", picture)
            cv2.waitKey(0)
            
      elif(Isrecognition_ON == False):
        draw_simple_rect(face_list,picture,pictureNumber)   


def draw_simple_rect(face_list, picture, pictureNumber):
     for (x, y, w, h) in face_list:
            
            # Draws a rectangle around the faces
            cv2.rectangle(picture, (x, y), (x+w, y+h), (255, 0, 0), 2)

     cv2.imshow("Faces found", picture)
     cv2.waitKey(0)

     # Prompts for label
     label = input("Enter a name: ")
        
     with open("labels.txt", "a") as f:
            f.write(f"{f'image_{pictureNumber}.png'} {label}\n")