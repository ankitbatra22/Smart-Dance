import json
import numpy as np

def l2_norm(ground_relative_coords, webcam_relative_coords):
	return np.linalg.norm(ground_relative_coords - webcam_relative_coords)

def compare_keypoints(ground_points, webcam_points):
	'''
	This function takes in pose landmarks of the form {'0': [x, y, z], '1': [x, y, z], ...} 
	for a ground truth video and a webcam video.
	'''
	ground_points_array = []
	webcam_points_array = []

	for i in range(len(ground_points)):
		ground_points_array.append(np.array(ground_points[str(i)])[0:2]) # only using x, y coordinates of keypoints
		webcam_points_array.append(np.array(webcam_points[str(i)])[0:2])

	ground_points_array = np.vstack(ground_points_array)
	webcam_points_array = np.vstack(webcam_points_array)

	return l2_norm(ground_relative_coords, webcam_relative_coords)