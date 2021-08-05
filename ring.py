import json
import getpass
from pathlib import Path
from pprint import pprint
import cv2
from ring_doorbell import Ring, Auth
from oauthlib.oauth2 import MissingTokenError
import time
import os
import imutils
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

# saving password and username for easier login 
cache_file = Path("test_token.cache")

# loading prereqs for facial recognition + detection
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read(), encoding="latin1")
le = pickle.loads(open("output/le.pickle", "rb").read(), encoding="latin1")

def token_updated(token):
    cache_file.write_text(json.dumps(token))


def otp_callback():
    auth_code = input("2FA code: ")
    return auth_code


def main():
    if cache_file.is_file():
        auth = Auth("RingProject/1.234e", json.loads(cache_file.read_text()), token_updated)
    else:
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        auth = Auth("RingProject/1.234e", None, token_updated)
        try:
            auth.fetch_token(username, password)
        except MissingTokenError:
            auth.fetch_token(username, password, otp_callback())

    ring = Ring(auth)
    ring.update_data()

    devices = ring.devices()

    id = -1
    current_id = None
    while True:
        try:
            ring.update_data()

        except:
            time.sleep(1)
            continue

        doorbell = devices['authorized_doorbots'][0]
        for event in doorbell.history(limit=20, kind='ding'):
            current_id = event['id']
            break

        if current_id != id:
            id = current_id

            handle = handle_video(ring)
        time.sleep(1)

def handle_video(ring):
    devices = ring.devices()
    doorbell = devices['authorized_doorbots'][0]
    doorbell.recording_download(
        doorbell.history(limit=100, kind='ding')[0]['id'],
        filename='new_ding.mp4',
        override=True)
    
    cap = cv2.VideoCapture('new_ding.mp4')

    #initialize counter 
    count = [0] * len(le.classes_)

    
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()

        #start facial detection + recognition

        # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.3:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]

                if (proba < 0.85):
                    name = "unknown"
                else:
                    name = le.classes_[j]
                    count[j] = count[j] + 1
                    if (count[j] == 10):
                        print(le.classes_[j]) 
                        print(" is here")

                # draw the bounding box of the face along with the associated probability
                if (name == "unknown"):
                    text = "{}".format(name)
                else:
                    text = "{}: {:.2f}%".format(name, proba * 100)

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


        #show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # cleanup
    cv2.destroyAllWindows()
    cap.release()

    
if __name__ == "__main__":
    main()