from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import json
import os
import pymongo

#Load configs
app = Flask(__name__)

with open("./configs/config.json") as dataFile:
  config = json.load(dataFile)

username, password = (config["username"], config["password"])

app.secret_key = "testing"

#DB Connection
client = pymongo.MongoClient(f'mongodb+srv://{username}:{password}@cluster0-xth9g.mongodb.net/Richard?retryWrites=true&w=majority')

# Mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        # Capture frame-by-frame
            success, image = camera.read()
            #print(success.shape)
            if not success:
                break
            else:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                print(results.pose_landmarks)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield (b'--image\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/train')
def train():
    print("hello")
    return "hello world"

@app.route('/analyze')
def analyze():
    return "analyze"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/about')
def about_us():
    return "about us"

@app.route('/info')
def info():
    return "info"

@app.route('/demo')
def demo():
    return "demo"

@app.route('/')
def index():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, port=8080)