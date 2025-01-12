import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import argparse
import yaml


def savePoseKITTI(results_file, pose):
    """
    Save camera pose in KITTI format to a results file.

    Parameters:
        results_file (str): Path to the output file where poses will be saved
        pose (np.ndarray): 4x4 transformation matrix representing camera pose

    Returns:
        None
    """
    with open(results_file, "a") as f:
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
    # Match features using the matcher
    matches = matcher.knnMatch(des1, des2, k=2)

    # Filter matches based on Lowe's ratio
    good_matches = [[m] for m, n in matches if m.distance < lowe_ratio * n.distance]

    # Check if there are enough good matches
    if len(good_matches) > 4:
        # Extract matched keypoints and descriptors
        matched_kp1 = [kp1[m[0].queryIdx] for m in good_matches]
        matched_kp2 = [kp2[m[0].trainIdx] for m in good_matches]
        des1_matched = np.float32([des1[m[0].queryIdx] for m in good_matches])
        des2_matched = np.float32([des2[m[0].trainIdx] for m in good_matches])
        return matched_kp1, matched_kp2, des1_matched, des2_matched, good_matches
    else:
        print("Insufficient good matches to estimate rotation.")
        return None, None, None, None, []


def consistent_feature_matching(
    current_kp_left,
    current_des_left,
    prev_kp_left,
    prev_des_left,
    prev_kp_right,
    prev_des_right,
    lowe_ratio,
    matcher,
):
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
        current_kp_left,
        prev_kp_left,
        current_des_left,
        prev_des_left,
        lowe_ratio,
        matcher,
    )
    # Check if there are any matches
    if not matches1:
        print("No consistent matches found between current_left and prev_left.")
        return None, None, None

    # Match prev_left and prev_right
    _, matched_prev_right_kp, _, _, matches2 = match_features(
        prev_kp_left, prev_kp_right, prev_des_left, prev_des_right, lowe_ratio, matcher
    )

    # Check if there are any matches
    if not matches2:
        print("No consistent matches found between prev_left and prev_right.")
        return None, None, None

    # Find consistent matches
    consistent_kp_current_left = []
    consistent_kp_prev_left = []
    consistent_kp_prev_right = []

    # Iterate over matches2
    for match2 in matches2:
        # Iterate over matches1
        for match1 in matches1:
            # Check if the matches are consistent
            if match2[0].queryIdx == match1[0].trainIdx:
                # Append the corresponding keypoints
                consistent_kp_current_left.append(
                    matched_current_left_kp[matches1.index(match1)]
                )
                consistent_kp_prev_left.append(
                    matched_prev_left_kp[matches1.index(match1)]
                )
                consistent_kp_prev_right.append(
                    matched_prev_right_kp[matches2.index(match2)]
                )

    if (
        not consistent_kp_current_left
        or not consistent_kp_prev_left
        or not consistent_kp_prev_right
    ):
        print("No consistent keypoints found across all images.")
        return None, None, None

    return consistent_kp_current_left, consistent_kp_prev_left, consistent_kp_prev_right


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file.

    Args:
        config_path (str, optional): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        dict: Dictionary containing the configuration parameters loaded from the YAML file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_odometry(dataset, config, results_file=None):
    """
    Compute visual odometry using stereo images.

    This function processes stereo image pairs to estimate the camera trajectory. It uses feature detection,
    matching, and triangulation to compute relative motion between frames. Configuration parameters are
    passed via a config dictionary rather than individual parameters.

    Args:
        dataset: Dataset object containing stereo image pairs
        config (dict): Configuration dictionary containing parameters like:
            - detector: Feature detector to use (SIFT, ORB, etc.)
            - matcher: Feature matcher to use (BF, FLANN)
            - real_time: Boolean for real-time visualization
            - figure_size: Size of visualization figure
        results_file (str, optional): Path to save trajectory results. Defaults to None.

    Returns:
        None
    """
    # Initialize detector based on config
    detector_name = config["detector"].upper()
    if detector_name == "SIFT":
        detector = cv2.SIFT_create()
    elif detector_name == "ORB":
        detector = cv2.ORB_create()
    elif detector_name == "BRISK":
        detector = cv2.BRISK_create()
    elif detector_name == "AKAZE":
        detector = cv2.AKAZE_create()
    elif detector_name == "KAZE":
        detector = cv2.KAZE_create()
    elif detector_name == "FAST":
        detector = cv2.FastFeatureDetector_create()
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")

    # Initialize matcher based on config
    matcher_name = config["matcher"].upper()
    if matcher_name == "BF":
        matcher = cv2.BFMatcher()
    elif matcher_name == "FLANN":
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    else:
        raise ValueError(f"Unsupported matcher: {matcher_name}")

    # Initialize variables
    prev_left_img, prev_right_img = None, None
    prev_kp_left, prev_des_left = None, None
    prev_kp_right, prev_des_right = None, None
    trajectory = [np.eye(4)]

    # Save the initial pose
    if results_file:
        savePoseKITTI(results_file, trajectory[-1])

    # Real-time visualization setup
    if config["real_time"]:
        fig = plt.figure(figsize=tuple(config["figure_size"]))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Estimated Camera Trajectory")

    # Iterate over the dataset
    for i, (current_left_img, current_right_img) in enumerate(dataset.gray):
        current_left_img = np.array(current_left_img)
        current_right_img = np.array(current_right_img)

        if i == 0:
            prev_left_img, prev_right_img = current_left_img, current_right_img
            continue

        # Get features and their descriptors for the left and right camera images
        current_kp_left, current_des_left = detector.detectAndCompute(
            current_left_img, None
        )
        current_kp_right, current_des_right = detector.detectAndCompute(
            current_right_img, None
        )

        if prev_kp_left is None:
            prev_kp_left, prev_des_left = detector.detectAndCompute(prev_left_img, None)
        if prev_kp_right is None:
            prev_kp_right, prev_des_right = detector.detectAndCompute(
                prev_right_img, None
            )

        # Match the features across 2 frames
        filt_current_left_kp, filt_prev_left_kp, filt_prev_right_kp = (
            consistent_feature_matching(
                current_kp_left,
                current_des_left,
                prev_kp_left,
                prev_des_left,
                prev_kp_right,
                prev_des_right,
                config["lowe_ratio"],
                matcher,
            )
        )

        if (
            filt_current_left_kp is None
            or filt_prev_left_kp is None
            or filt_prev_right_kp is None
        ):
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
            if (
                disparities[j] > config["min_disparity"]
                and epipolar_errors[j] < config["max_epipolar_error"]
            ):
                matches0.append(filt_current_left_kp[j])
                matches0_prev.append(filt_prev_left_kp[j])
                matches1_prev.append(filt_prev_right_kp[j])
            else:
                fails.append(filt_prev_left_kp[j])

        image_points = keypoints_to_np(matches0)
        prev_2d_points_left = keypoints_to_np(matches0_prev)
        prev_2d_points_right = keypoints_to_np(matches1_prev)

        # Get 3d points from the 2 stereo images from the previous timestep
        prev_3d_points = cv2.triangulatePoints(
            dataset.calib.P_rect_00,
            dataset.calib.P_rect_10,
            prev_2d_points_left.T,
            prev_2d_points_right.T,
        )

        # Convert 3D points from homogeneous coordinates
        prev_3d_points = cv2.convertPointsFromHomogeneous(prev_3d_points.T).reshape(
            -1, 3
        )

        # Use PnP with RANSAC for roto-translation estimation
        if prev_3d_points.shape[0] > 3:
            print(f"Solving PnP RANSAC for {prev_3d_points.shape[0]} points")
            _, rvec, translation_vector, _ = cv2.solvePnPRansac(
                prev_3d_points,
                image_points,
                dataset.calib.K_cam0,
                None,
                flags=getattr(cv2, config["pnp_method"]),
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
        if config["real_time"]:
            positions = np.array([pose[:3, 3] for pose in trajectory])

            ax.clear()
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                label="Camera Trajectory",
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Estimated Camera Trajectory")
            ax.legend()
            plt.pause(0.01)  # Pause to update the plot in real-time

        print(f"Frame {i}: Pose: {trajectory[-1]}")
        prev_left_img, prev_right_img = current_left_img, current_right_img
        prev_kp_left, prev_des_left = current_kp_left, current_des_left
        prev_kp_right, prev_des_right = current_kp_right, current_des_right

    return trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process KITTI dataset for visual odometry."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is None:
        # Get config from same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(script_dir, "config.yaml")
    config = load_config(args.config)

    # Use config values
    basedir = config["basedir"]
    sequence = config["sequence"]
    experiment = str(config["experiment"])
    results_dir = config["results_dir"]

    # Create results directory if it doesn't exist
    results_dir = os.path.join(results_dir, sequence)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create results file if it doesn't exist
    results_file = os.path.join(results_dir, f"{sequence}_{experiment}.txt")
    if not os.path.exists(results_file):
        print(f"Results file {sequence}.txt does not exist. Creating it...")
        with open(results_file, "w") as f:
            f.write("")

    dataset = pykitti.odometry(basedir, sequence)

    trajectory = compute_odometry(dataset, config, results_file=results_file)

    # Plot the trajectory if real-time is not enabled
    if not config["real_time"]:
        positions = np.array([pose[:3, 3] for pose in trajectory])
        fig = plt.figure(figsize=tuple(config["figure_size"]))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label="Camera Trajectory",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Estimated Camera Trajectory")
        ax.legend()
    plt.show()
