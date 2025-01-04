import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import argparse

def savePoseKITTI(results_file, pose):
    with open(results_file, 'a') as f:
        f.write(f"{' '.join(f'{x:.6e}' for x in pose[:3,:4].flatten())}\r\n")

def keypoints_to_np(keypoints):
    """
    Converts a list of OpenCV keypoints to a NumPy array of their coordinates.

    Parameters:
        keypoints (list of cv2.KeyPoint): List of OpenCV keypoints.

    Returns:
        np.ndarray: Array of shape (N, 2) containing keypoint coordinates.
    """

    return np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)


def match_features(kp1, kp2, des1, des2, lowe_ratio=0.75, matcher=cv2.BFMatcher()):
    """
    Matches features between two sets of keypoints and descriptors using the BFMatcher.

    Parameters:
        kp1 (list of cv2.KeyPoint): Keypoints from the first image.
        kp2 (list of cv2.KeyPoint): Keypoints from the second image.
        des1 (np.ndarray): Descriptors from the first image.
        des2 (np.ndarray): Descriptors from the second image.
        lowe_ratio (float): Lowe's ratio for filtering good matches.
        matcher (cv2.BFMatcher): Matcher to use for feature matching.

    Returns:
        tuple: Matched keypoints and descriptors for both images and the matches list.
    """

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [[m] for m, n in matches if m.distance < lowe_ratio * n.distance]

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
                                prev_kp_right, prev_des_right, lowe_ratio, matcher):
    
    
    matched_current_left_kp, matched_prev_left_kp, _, _, matches1 = match_features(
        current_kp_left, prev_kp_left, current_des_left, prev_des_left, lowe_ratio, matcher
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
        lowe_ratio (float): Lowe's ratio for filtering good matches.

    Returns:
        tuple: Consistent keypoints from the current left, previous left, and previous right images.
    """

    # Match current_left and prev_left
    matched_current_left_kp, matched_prev_left_kp, _, _, matches1 = match_features(
        current_kp_left, prev_kp_left, current_des_left, prev_des_left, lowe_ratio, matcher
    )
    
    if not matches1:
        print("No consistent matches found between current_left and prev_left.")
        return None, None, None
    
    # Match prev_left and prev_right
    _, matched_prev_right_kp, _, _, matches2 = match_features(
        prev_kp_left, prev_kp_right, prev_des_left, prev_des_right, lowe_ratio, matcher
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


def compute_odometry(dataset, real_time, lowe_ratio=0.75, results_file=None):
    """
    Computes the odometry trajectory for a dataset using feature matching and triangulation.

    Parameters:
        dataset (pykitti.odometry): KITTI odometry dataset instance.
        online (bool): If True, performs real-time visualization of the trajectory.
        lowe_ratio (float): Lowe's ratio for feature matching.

    Returns:
        list: List of transformation matrices representing the trajectory.
    """

    # Initialize detector
    # detector = cv2.ORB_create()
    detector = cv2.SIFT_create()
    # detector = cv2.xfeatures2d.SURF_create()
    # detector = cv2.BRISK_create()
    # detector = cv2.AKAZE_create()
    # detector = cv2.KAZE_create()
    # detector = cv2.FastFeatureDetector_create()
    
    # Initialize matcher
    matcher = cv2.BFMatcher()
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    
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
        current_kp_left, current_des_left = detector.detectAndCompute(current_left_img, None)
        current_kp_right, current_des_right = detector.detectAndCompute(current_right_img, None)

        if prev_kp_left is None:
            prev_kp_left, prev_des_left = detector.detectAndCompute(prev_left_img, None)
        if prev_kp_right is None:
            prev_kp_right, prev_des_right = detector.detectAndCompute(prev_right_img, None)

        # Match the features across 2 frames
        filt_current_left_kp, filt_prev_left_kp, filt_prev_right_kp = consistent_feature_matching(
            current_kp_left, current_des_left, prev_kp_left, prev_des_left,
            prev_kp_right, prev_des_right, lowe_ratio, matcher
        )
        
        if filt_current_left_kp is None or filt_prev_left_kp is None or filt_prev_right_kp is None:
            print(f"Skipping frame {i} due to insufficient consistent keypoints.")
            continue

        # Checking epipolar constraints
        epipolar_errors = []
        for j in range(len(filt_prev_left_kp)):
            pt0 = filt_prev_left_kp[j].pt
            pt1 = filt_prev_right_kp[j].pt
            error = abs(pt0[1] - pt1[1])
            epipolar_errors.append(error)
        
        # Checking disparity constraints
        disparities = []
        for k in range(len(filt_prev_left_kp)):
            pt0 = filt_prev_left_kp[k].pt
            pt1 = filt_prev_right_kp[k].pt
            disparity = pt0[0] - pt1[0]
            disparities.append(disparity)
        
        matches0 = []
        matches0_prev = []
        matches1_prev = []
        fails = []
        
        # Checking if disparity is positive and epipolar error is less than 1.0
        for j in range(len(filt_prev_left_kp)):
            if disparities[j] > 0 and epipolar_errors[j] < 1.0:
                matches0.append(filt_current_left_kp[j])
                matches0_prev.append(filt_prev_left_kp[j])
                matches1_prev.append(filt_prev_right_kp[j])
            else:
                fails.append(filt_prev_left_kp[j])

        image_points = keypoints_to_np(matches0)
        prev_2d_points_left = keypoints_to_np(matches0_prev)
        prev_2d_points_right = keypoints_to_np(matches1_prev)


        # image_points = keypoints_to_np(filt_current_left_kp)
        # prev_2d_points_left = keypoints_to_np(filt_prev_left_kp)
        # prev_2d_points_right = keypoints_to_np(filt_prev_right_kp)

        # Get 3d points from the 2 stereo images from the previous timestep
        prev_3d_points = cv2.triangulatePoints(
            dataset.calib.P_rect_00, dataset.calib.P_rect_10,
            prev_2d_points_left.T, prev_2d_points_right.T
        )
        prev_3d_points = cv2.convertPointsFromHomogeneous(prev_3d_points.T).reshape(-1, 3)

        # Use PnP with RANSAC for roto-translation estimation
        if prev_3d_points.shape[0] > 3:
            print(f"Solving PnP RANSAC for {prev_3d_points.shape[0]} points")
            _, rvec, translation_vector, _ = cv2.solvePnPRansac(
                prev_3d_points, image_points, dataset.calib.K_cam0, None, flags=cv2.USAC_MAGSAC
            )
        else:
            print("PnP RANSAC failed")
            continue

        # Store the estimate as a transformation matrix
        r_matrix = cv2.Rodrigues(rvec)[0]
        transformation = np.eye(4)
        transformation[:3, :3] = r_matrix
        transformation[:3, 3] = translation_vector.T

        trajectory.append(trajectory[-1] @ np.linalg.inv(transformation))

        if results_file:
            savePoseKITTI(results_file, trajectory[-1])

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
    parser.add_argument('--frames', type=int, default=None, help="Number of frames to process. Default is full sequence")
    args = parser.parse_args()

    # Specify the sequence to load
    sequence = '00'
    experiment = '2'
    
    # Path to the KITTI dataset
    basedir = "/home/patweatharva/IFROS/rspa/tdv_lab/data_odometry_gray/dataset"
    results_dir = f"/home/patweatharva/IFROS/rspa/tdv_lab/data_odometry_gray/dataset/results/{sequence}"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir, f"{sequence}_{experiment}.txt")
    if not os.path.exists(results_file):
        print(f"Results file {sequence}.txt does not exist. Creating it...")
        with open(results_file, 'w') as f:
            f.write("")
            

    # Load the KITTI dataset
    if args.frames is not None:
        frames = [i for i in range(args.frames)]
    else:
        frames = None
    dataset = pykitti.odometry(basedir, sequence, frames=frames)

    trajectory = compute_odometry(dataset, args.real_time, lowe_ratio=0.75, results_file=results_file)

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
        plt.show()  # This will display the plot interactively
