#Use this script to generate gifs and extract keypoints from videos. 
#Ensure the video you're working with exists in the 'videos' folder 
#and specify the name of the video on line 16 in video_name.

import cv2
import mediapipe as mp
import numpy as np
import imageio
import pickle
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_name = 'renegade'
video_path = 'keypoint-extractor/videos/' + video_name
cap = cv2.VideoCapture(video_path)
i=0
temp_path = 'keypoint-extractor/temp/annotated_image'
output_path = 'keypoint-extractor/output/'
annotated_frames = []
keypoints_dict = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    if results.pose_landmarks is not None:
	    annotated_pose_landmarks = {str(j): [lmk.x, lmk.y, lmk.z] for j, lmk in enumerate(results.pose_landmarks.landmark)}
	    keypoints_dict.append(annotated_pose_landmarks)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    cv2.imwrite(temp_path + str(i) + '.png', image)
    annotated_frames.append(temp_path + str(i) + '.png')
    i+=1
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()


def frames_to_gif(frames, output_name):
	images = []
	for frame in frames:
		images.append(imageio.imread(frame))
	imageio.mimsave(output_path + output_name + '.gif', images)
	print("Gif saved.")

frames_to_gif(annotated_frames, video_name)


with open(output_path + video_name + '-keypoints.json', 'w') as fp:
    json.dump(keypoints_dict, fp)
