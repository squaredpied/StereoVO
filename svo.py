import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import argparse

def keypoints_to_np(keypoints):
    """
    Converts a list of OpenCV keypoints to a NumPy array of their coordinates.

    Parameters:
        keypoints (list of cv2.KeyPoint): List of OpenCV keypoints.

    Returns:
        np.ndarray: Array of shape (N, 2) containing keypoint coordinates.
    """

    return np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)


def match_features(kp1, kp2, des1, des2, ratio):
    """
    Matches features between two sets of keypoints and descriptors using the BFMatcher.

    Parameters:
        kp1 (list of cv2.KeyPoint): Keypoints from the first image.
        kp2 (list of cv2.KeyPoint): Keypoints from the second image.
        des1 (np.ndarray): Descriptors from the first image.
        des2 (np.ndarray): Descriptors from the second image.
        ratio (float): Lowe's ratio for filtering good matches.

    Returns:
        tuple: Matched keypoints and descriptors for both images and the matches list.
    """

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [[m] for m, n in matches if m.distance < ratio * n.distance]

    if len(good_matches) > 4:
        matched_kp1 = [kp1[m[0].queryIdx] for m in good_matches]
        matched_kp2 = [kp2[m[0].trainIdx] for m in good_matches]
        des1_matched = np.float32([des1[m[0].queryIdx] for m in good_matches])
        des2_matched = np.float32([des2[m[0].trainIdx] for m in good_matches])
        return matched_kp1, matched_kp2, des1_matched, des2_matched, good_matches
    else:
        print("Insufficient good matches to estimate rotation.")
        return None, None, None, None, []


def consistent_feature_matching(current_kp_left, current_des_left, prev_kp_left, prev_des_left, 
                                prev_kp_right, prev_des_right, ratio):
    matched_current_left_kp, matched_prev_left_kp, _, _, matches1 = match_features(
        current_kp_left, prev_kp_left, current_des_left, prev_des_left, ratio
    )
    """
    Ensures consistency in feature matching across multiple frames.

    Parameters:
        current_kp_left (list of cv2.KeyPoint): Keypoints in the current left image.
        current_des_left (np.ndarray): Descriptors in the current left image.
        prev_kp_left (list of cv2.KeyPoint): Keypoints in the previous left image.
        prev_des_left (np.ndarray): Descriptors in the previous left image.
        prev_kp_right (list of cv2.KeyPoint): Keypoints in the previous right image.
        prev_des_right (np.ndarray): Descriptors in the previous right image.
        ratio (float): Lowe's ratio for filtering good matches.

    Returns:
        tuple: Consistent keypoints from the current left, previous left, and previous right images.
    """

    # Match current_left and prev_left
    matched_current_left_kp, matched_prev_left_kp, _, _, matches1 = match_features(
        current_kp_left, prev_kp_left, current_des_left, prev_des_left, ratio
    )
    
    if not matches1:
        print("No consistent matches found between current_left and prev_left.")
        return None, None, None
    
    # Match prev_left and prev_right
    _, matched_prev_right_kp, _, _, matches2 = match_features(
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


def compute_odometry(dataset, real_time, sift_ratio=0.75):
    """
    Computes the odometry trajectory for a dataset using feature matching and triangulation.

    Parameters:
        dataset (pykitti.odometry): KITTI odometry dataset instance.
        online (bool): If True, performs real-time visualization of the trajectory.
        sift_ratio (float): Lowe's ratio for SIFT feature matching.

    Returns:
        list: List of transformation matrices representing the trajectory.
    """

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Initialize variables
    prev_left_img, prev_right_img = None, None
    prev_kp_left, prev_des_left = None, None
    prev_kp_right, prev_des_right = None, None
    trajectory = [np.eye(4)]

    if real_time:
        # Real-time visualization setup
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Estimated Camera Trajectory')

    for i, (current_left_img, current_right_img) in enumerate(dataset.gray):
        current_left_img = np.array(current_left_img)
        current_right_img = np.array(current_right_img)

        if i == 0:
            prev_left_img, prev_right_img = current_left_img, current_right_img
            continue
        
        # Get features and their descriptors for the left and right camera images
        current_kp_left, current_des_left = sift.detectAndCompute(current_left_img, None)
        current_kp_right, current_des_right = sift.detectAndCompute(current_right_img, None)

        if prev_kp_left is None:
            prev_kp_left, prev_des_left = sift.detectAndCompute(prev_left_img, None)
        if prev_kp_right is None:
            prev_kp_right, prev_des_right = sift.detectAndCompute(prev_right_img, None)

        # Match the features across 2 frames
        filt_current_left_kp, filt_prev_left_kp, filt_prev_right_kp = consistent_feature_matching(
            current_kp_left, current_des_left, prev_kp_left, prev_des_left,
            prev_kp_right, prev_des_right, sift_ratio
        )

        if filt_current_left_kp is None or filt_prev_left_kp is None or filt_prev_right_kp is None:
            print(f"Skipping frame {i} due to insufficient consistent keypoints.")
            continue

        image_points = keypoints_to_np(filt_current_left_kp)
        prev_2d_points_left = keypoints_to_np(filt_prev_left_kp)
        prev_2d_points_right = keypoints_to_np(filt_prev_right_kp)

        # Get 3d points from the 2 stereo images from the previous timestep
        prev_3d_points = cv2.triangulatePoints(
            dataset.calib.P_rect_00, dataset.calib.P_rect_10,
            prev_2d_points_left.T, prev_2d_points_right.T
        )
        prev_3d_points = cv2.convertPointsFromHomogeneous(prev_3d_points.T).reshape(-1, 3)

        # Use PnP with RANSAC for roto-translation estimation
        _, rvec, translation_vector, _ = cv2.solvePnPRansac(
            prev_3d_points, image_points, dataset.calib.K_cam0, None
        )

        # Store the estimate as a transformation matrix
        r_matrix = cv2.Rodrigues(rvec)[0]
        transformation = np.eye(4)
        transformation[:3, :3] = r_matrix
        transformation[:3, 3] = translation_vector.T

        trajectory.append(trajectory[-1] @ np.linalg.inv(transformation))

        # Plot the trajectory after processing each frame
        if real_time:
            positions = np.array([pose[:3, 3] for pose in trajectory])

            ax.clear()
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Camera Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Estimated Camera Trajectory')
            ax.legend()
            plt.pause(0.01)  # Pause to update the plot in real-time

        print(f"Frame {i}: Pose: {trajectory[-1]}")
        prev_left_img, prev_right_img = current_left_img, current_right_img
        prev_kp_left, prev_des_left = current_kp_left, current_des_left
        prev_kp_right, prev_des_right = current_kp_right, current_des_right

    return trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KITTI dataset for visual odometry.")
    parser.add_argument('--real_time', action='store_true', help="Run the script in real time mode.")
    args = parser.parse_args()

    # Path to the KITTI dataset
    basedir = "C:/Users/prec1/Documents/fer/rspa/perception/data_odometry_gray/dataset"

    # Specify the sequence to load
    sequence = '00'

    # Load the KITTI dataset
    dataset = pykitti.odometry(basedir, sequence)

    trajectory = compute_odometry(dataset, args.real_time)

    if not args.real_time:
        positions = np.array([pose[:3, 3] for pose in trajectory])
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Camera Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Estimated Camera Trajectory')
        ax.legend()
    plt.show()
