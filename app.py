from flask import Flask, render_template, Response, session, request, redirect, url_for, flash
import cv2
import mediapipe as mp
import json
import os
import pymongo
from flask_pymongo import PyMongo
import bcrypt
import numpy as np

#Load configs
app = Flask(__name__)

with open("./configs/config.json") as dataFile:
  config = json.load(dataFile)

username, password = (config["username"], config["password"])
app.secret_key = "testing"
app.config['MONGO_DBNAME'] = 'dance-assist'
app.config['MONGO_URI'] = f'mongodb+srv://{username}:{password}@cluster0.vdghb.mongodb.net/dance-assist'

mongo = PyMongo(app)

#DB Connection
#lient = pymongo.MongoClient(f'mongodb+srv://{username}:{password}@cluster0-xth9g.mongodb.net/Richard?retryWrites=true&w=majority')

#db = client.get_database('total_records')
#records = db.register

# Mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def connect_points(points, image):
    points_connect_dict = {
        1: [2, 0],
        2: [1, 3],
        3: [7],
        4: [0, 5],
        5: [4, 6],
        6: [8, 5],
        7: [3],
        8: [6],
        9: [10],
        10: [9],
        11: [12, 13],
        12: [11, 14],
        13: [11, 15],
        14: [12, 16],
        15: [21, 17, 13],
        16: [22, 20, 18, 14],
        17: [15, 19],
        18: [20, 16],
        19: [17, 15],
        20: [18, 16],
        21: [15],
        22: [16],
        23: [11, 24, 25],
        24: [23, 26, 12],
        25: [23, 27],
        26: [24, 28],
        27: [25, 31, 29],
        28: [30, 32, 26],
        29: [27, 31],
        30: [28, 32],
        31: [29, 27]
    }

    for p in points_connect_dict:
        curr_point = points[p]
        for endpoint in points_connect_dict[p]:
            endpoint = points[endpoint]
            cv2.line(image, curr_point, endpoint, (np.random.randint))




def gen_frames():
    camera = cv2.VideoCapture(0)
    # infinite webcam loop
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

                # results.pose_landmarks gives us the keypoints, 28 * 2 *2 
                # results.pose_keypoints_score gives us the confidence of the keypoints
                # results.pose_keypoints_score[0] gives us the confidence of the first keypoint
                # results.pose_keypoints_score[0][0] gives us the confidence of the first keypoint of the first person
                # results.pose_keypoints_score[0][1] gives us the confidence of the second keypoint of the first person
                # results.pose_keypoints_score[0][2] gives us the confidence of the third keypoint of the first person

                #print(results.pose_landmarks)
                #print(results.pose_keypoints_score)
                #print(results.pose_keypoints_score[0])



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

@app.route('/home')
def home():
    return render_template('home.html', username=session['username'])

@app.route('/')
def index():
    if 'username' in session:
        return render_template('home.html', name=session['username'])
        return 'You are logged in as ' + session['username']

    return render_template('index.html')
    

@app.route('/login', methods=['POST'])
def login():
    users = mongo.db.users
    login_user = users.find_one({'name': request.form['username']})

    if login_user:
        if bcrypt.hashpw(request.form['pass'].encode('utf-8'), login_user['password']) == login_user['password']:
            session['username'] = request.form['username']
            return redirect(url_for('index'))

    return 'Invalid username or password'

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'name' : request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'name':request.form['username'], 'password': hashpass})
            session['username'] =  request.form['username']
            return redirect(url_for('index'))

        return 'That username already exists!'

    return render_template('register.html')

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True)