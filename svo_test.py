import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pykitti

def keypoints_to_np(keypoints):
    """
    Converts a list of cv2.KeyPoint objects to a NumPy array of shape (N, 2).
    
    Parameters:
        keypoints (list of cv2.KeyPoint): The keypoints to convert.
        
    Returns:
        np.ndarray: NumPy array of shape (N, 2) containing keypoint coordinates.
    """
    return np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)

def match_features(kp1, kp2, des1, des2, ratio):
    matched_kp1, matched_kp2 = [], []
    des1_matched, des2_matched = [], []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])

    if len(good_matches) > 4:
        matched_kp1 = [kp1[m[0].queryIdx] for m in good_matches]
        matched_kp2 = [kp2[m[0].trainIdx] for m in good_matches]

        des1_matched = np.float32([des1[m[0].queryIdx] for m in good_matches])
        des2_matched = np.float32([des2[m[0].trainIdx] for m in good_matches])
    else:
        print("Insufficient good matches to estimate roto")

    return matched_kp1, matched_kp2, des1_matched, des2_matched, good_matches

def consistent_feature_matching(current_kp_left, current_des_left, prev_kp_left, prev_des_left, 
                                prev_kp_right, prev_des_right, ratio):
    """
    Matches features and ensures consistency across current_left, prev_left, and prev_right images.

    Returns:
        consistent_kp_current_left (list): Consistent keypoints in the current left image.
        consistent_kp_prev_left (list): Consistent keypoints in the previous left image.
        consistent_kp_prev_right (list): Consistent keypoints in the previous right image.
    """
    # Match current_left and prev_left
    matched_current_left_kp, matched_prev_left_kp, _, _, matches1 = match_features(
        current_kp_left, prev_kp_left, current_des_left, prev_des_left, ratio
    )
    
    if not matches1:
        print("No consistent matches found between current_left and prev_left.")
        return None, None, None
    
    # Match prev_left and prev_right
    matched_prev_left_kp_2, matched_prev_right_kp, _, _, matches2 = match_features(
        prev_kp_left, prev_kp_right, prev_des_left, prev_des_right, ratio
    )

    if not matches2:
        print("No consistent matches found between prev_left and prev_right.")
        return None, None, None
    
    # Find consistent matches
    consistent_kp_current_left = []
    consistent_kp_prev_left = []
    consistent_kp_prev_right = []

    for match2 in matches2:
        for match1 in matches1:
            if match2[0].queryIdx == match1[0].trainIdx:
                # Append the corresponding keypoints
                consistent_kp_current_left.append(matched_current_left_kp[matches1.index(match1)])
                consistent_kp_prev_left.append(matched_prev_left_kp[matches1.index(match1)])
                consistent_kp_prev_right.append(matched_prev_right_kp[matches2.index(match2)])
    
    if not consistent_kp_current_left or not consistent_kp_prev_left or not consistent_kp_prev_right:
        print("No consistent keypoints found across all images.")
        return None, None, None
    
    return consistent_kp_current_left, consistent_kp_prev_left, consistent_kp_prev_right

# Path to the KITTI dataset
basedir = "C:/Users/prec1/Documents/fer/rspa/perception/data_odometry_gray/dataset"

# Specify the sequence to load
sequence = '00'

# Load the KITTI dataset
dataset = pykitti.odometry(basedir, sequence)
calibration = dataset.calib
print(calibration)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Initialize variables
prev_left_img = None
prev_right_img = None
sift_ratio = 0.75

# Initialize pose as an identity matrix
trajectory = [np.eye(4)]

# Iterate over the grayscale image pairs in the dataset
for i, (current_left_img, current_right_img) in enumerate(dataset.gray):
    # Convert PIL images to NumPy arrays
    current_left_img = np.array(current_left_img)
    current_right_img = np.array(current_right_img)

    if i == 0:
        prev_left_img = current_left_img
        prev_right_img = current_right_img
        continue

    # Detect SIFT keypoints in the grayscale images
    current_kp_left, current_des_left = sift.detectAndCompute(current_left_img, None)
    prev_kp_left, prev_des_left = sift.detectAndCompute(prev_left_img, None)
    prev_kp_right, prev_des_right = sift.detectAndCompute(prev_right_img, None)

    # Match and filter consistent features
    filt_current_left_kp, filt_prev_left_kp, filt_prev_right_kp = consistent_feature_matching(
        current_kp_left, current_des_left, prev_kp_left, prev_des_left, 
        prev_kp_right, prev_des_right, sift_ratio
    )

    if filt_current_left_kp is None or filt_prev_left_kp is None or filt_prev_right_kp is None:
        print(f"Skipping frame {i} due to insufficient consistent keypoints.")
        continue

    # Convert keypoints to NumPy arrays
    image_points = keypoints_to_np(filt_current_left_kp)
    prev_2d_points_left = keypoints_to_np(filt_prev_left_kp)
    prev_2d_points_right = keypoints_to_np(filt_prev_right_kp)

    # Perform triangulation
    prev_3d_points = cv2.triangulatePoints(
        dataset.calib.P_rect_00, dataset.calib.P_rect_10,
        prev_2d_points_left.T, prev_2d_points_right.T
    )
    prev_3d_points = cv2.convertPointsFromHomogeneous(prev_3d_points.T).reshape(-1, 3)

    # Motion estimation
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        prev_3d_points, image_points, dataset.calib.K_cam0, None
    )

    r_matrix = cv2.Rodrigues(rvec)[0]

    # Construct homogeneous transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = r_matrix
    transformation[:3, 3] = translation_vector.T

    # Update cumulative pose
    trajectory.append(trajectory[-1] @ np.linalg.inv(transformation))

    prev_left_img = current_left_img
    prev_right_img = current_right_img

    # Debug visualization
    print(f"Frame {i}: Pose: {trajectory[-1]}")

# Extract camera positions for visualization
positions = np.array([pose[:3, 3] for pose in trajectory])

# Visualize the trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Camera Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Estimated Camera Trajectory')
ax.legend()
plt.show()
