from flask import Flask, render_template, Response, session, request, redirect, url_for, flash
import cv2
import mediapipe as mp
import json
import os
import pymongo
from flask_pymongo import PyMongo
import bcrypt
import numpy as np
import os

# Load configs
app = Flask(__name__)

with open("./configs/config.json") as dataFile:
    config = json.load(dataFile)

username, password = (config["username"], config["password"])
app.secret_key = "testing"
app.config['MONGO_DBNAME'] = 'dance-assist'
app.config['MONGO_URI'] = f'mongodb+srv://{username}:{password}@cluster0.vdghb.mongodb.net/dance-assist'

mongo = PyMongo(app)

# Mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def l2_norm(ground_relative_coords, webcam_relative_coords):
	return np.linalg.norm(ground_relative_coords - webcam_relative_coords)

def print_data(ground_points, webcam_points, translation_factors, w, h):
    print(ground_points[str(11)][0:2] * np.array([w, h]) - np.array(list(translation_factors)))
    print(webcam_points[11][0:2]* np.array([w, h]))

def compare_keypoints(ground_points, webcam_points, w, h, translation_factors):
	ground_points_array = []
	webcam_points_array = []
    
	for i in range(len(ground_points)):
		ground_points_array.append(np.array(ground_points[str(i)])[0:2]* np.array([w, h]) - 
        np.array(list(translation_factors))) # only using x, y coordinates of keypoints

		webcam_points_array.append(np.array(webcam_points[i])[0:2]* np.array([w, h]))

	ground_points_array = np.vstack(ground_points_array)
	webcam_points_array = np.vstack(webcam_points_array)

	return l2_norm(ground_points_array, webcam_points_array)

def connect_points(points, translation_factors, image, image_shape, scale):
    h, w = image_shape
    points_connect_dict = {
        1: [2, 0],
        2: [3],
        3: [7],
        4: [0, 5],
        5: [6],
        6: [8],
        9: [10],
        11: [13],
        12: [11, 14],
        13: [15],
        14: [16],
        15: [21],
        16: [20, 14],
        17: [15],
        18: [20, 16],
        19: [17],
        20: [16],
        22: [16],
        23: [11, 25],
        24: [23, 12],
        25: [27],
        26: [24, 28],
        27: [31, 29],
        28: [30, 32],
        29: [31],
        30: [32],
        32: [28],
    }
    for p in points_connect_dict:
        curr_point = points[str(p)][0:2] * np.array([w, h]) - \
            np.array(list(translation_factors))

        for endpoint in points_connect_dict[p]:
            endpoint = points[str(endpoint)][0:2] * np.array([w, h]) - \
                np.array(list(translation_factors))

            cv2.line(image, (round(curr_point[0] * scale), round(curr_point[1] * scale)),
                     (round(endpoint[0] * scale), round(endpoint[1] * scale)), (0, 0, 255), thickness=10)

    return image


def get_translation_factor(gt, person, h, w):
    # use point 11 and 12

    x_gt, y_gt = gt['11'][0]*w, gt['11'][1]*h
    x_person, y_person = person[11][0]*w, person[11][1]*h

    if x_person >= x_gt:
        return x_person - x_gt, y_person - y_gt
    elif x_person <= x_gt:
        return x_gt - x_person, y_gt - y_person


def put_text(image, text, h, w):
    image = cv2.putText(img=image, org=(w - 700, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), text=text, 
    thickness= 3)
    return image


def gen_frames():
    with open('metadata/' + "formatted-mustafa-dance-keypoints" + '.json') as f:
        data = json.load(f)

    counter = 0
    camera = cv2.VideoCapture(0)
    
    counter_start = 1
    max_count = 161
    while True and counter_start <= (max_count - 1):
        counter_start  += 1
        success, image = camera.read()
        if not success:
            pass
        else:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            h, w, _ = image.shape

            
            image = cv2.putText(img=image, org=(w//2, h//2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), 
            text= str((max_count - counter_start)//40), thickness=2)
       
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--image\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

# infinite webcam loop
    avg = []
    while True:
        if counter == len(data) - 1:
            counter = 0
        else:
            counter += 1

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            # Capture frame-by-frame
            success, image = camera.read()

            # print(success.shape)
            if not success:
                break
            else:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                groop = image

                image.flags.writeable = False
                results = pose.process(image)

                pose_landmarks = results.pose_landmarks
                if pose_landmarks is not None:
                    
                    scale_t = 1.0
                    h, w, _ = image.shape
                    pose_landmarks_str_keys = {str(i): [lmk.x, lmk.y, lmk.z]
                                      for i, lmk in enumerate(pose_landmarks.landmark)}
                    pose_landmarks = {i: [lmk.x, lmk.y, lmk.z]
                                      for i, lmk in enumerate(pose_landmarks.landmark)}

                    data[counter] =  {i:[scale_t*coord for coord in data[counter][i]] for i in data[counter]}

                    x_t, y_t = get_translation_factor(
                        data[counter], pose_landmarks, h, w)

                    image.flags.writeable = True

                    # results.pose_landmarks gives us the keypoints, 28 * 2 *2
                    # results.pose_keypoints_score gives us the confidence of the keypoints
                    # results.pose_keypoints_score[0] gives us the confidence of the first keypoint
                    # results.pose_keypoints_score[0][0] gives us the confidence of the first keypoint of the first person
                    # results.pose_keypoints_score[0][1] gives us the confidence of the second keypoint of the first person
                    # results.pose_keypoints_score[0][2] gives us the confidence of the third keypoint of the first person

                    # print(results.pose_landmarks)
                    # print(results.pose_keypoints_score)
                    # print(results.pose_keypoints_score[0])

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    thingy = connect_points(
                        data[counter], (x_t, y_t), groop, (h, w), scale=1.0)
                    
                    thingy = cv2.cvtColor(thingy, cv2.COLOR_RGB2BGR)
                    thingy = connect_points(pose_landmarks_str_keys, (0, 0), thingy, (h, w), scale=0.1)

                    points = compare_keypoints(data[counter], pose_landmarks, w, h, (x_t, y_t))
                    
                    if len(avg) == 20:
                        avg.pop(0)
                        avg.append(points)
                    else:
                        avg.append(points)


                    thingy = put_text(thingy, "Score :" + str(round(sum(avg)/len(avg), 2)), h, w)
                    ret, buffer = cv2.imencode('.jpg', thingy)
                    image = buffer.tobytes()
                    yield (b'--image\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/dance/<song>')
def dance(song):
    global song_name
    song_name = song
    return render_template('dance.html', song=song)


@app.route('/train')
def train():
    numDir = len(os.listdir("./metadata"))
    numFiles = os.listdir("./metadata")
    dances = ["Renegade", "Outwest", "Carlton", "SavageLove", "Floss", "SaySo"]
    return render_template('dancepage.html', numDir=numDir, numFiles=numFiles, dances=dances)


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
        return render_template('home.html', username=session['username'])
        return 'You are logged in as ' + session['username']

    return render_template('ind.html')


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
        existing_user = users.find_one({'name': request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(
                request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert(
                {'name': request.form['username'], 'password': hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('index'))

        return 'That username already exists!'

    return render_template('register.html')


if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True)
