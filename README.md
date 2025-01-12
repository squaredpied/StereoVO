# Stereo Visual Odometry

This project implements a stereo visual odometry system using the KITTI dataset. It estimates camera trajectory by processing consecutive stereo image pairs using feature detection, matching, and triangulation.

## Features

- Multiple feature detector options (SIFT, ORB, BRISK, AKAZE, KAZE, FAST)
- Feature matching with ratio test
- Epipolar geometry constraints
- 3D point triangulation
- PnP with RANSAC for pose estimation
- Real-time visualization option
- Configurable parameters via YAML file

## Usage

1. Clone the repository:
```bash
git clone https://github.com/squaredpied/StereoVO.git
cd StereoVO
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the KITTI dataset and organize it according to the structure described in the Dataset Structure section.

5. Configure your settings:
   - Edit `config.yaml` to set your dataset path and desired parameters

6. Run the visual odometry system:
```bash
python svo.py --config config.yaml
```
The script will:
1. Process stereo image pairs from the specified sequence
2. Estimate camera trajectory
3. Save results in the specified results directory
4. Display trajectory visualization (real-time or at the end)
## Dataset Structure

The code expects the KITTI dataset in the following structure:

```
data_odometry_gray/dataset/
├── sequences/
│   ├── 00/
│   │   ├── image_0/      # Left camera images
│   │   ├── image_1/      # Right camera images
│   │   ├── times.txt     # Timestamps for each frame
│   │   └── calib.txt     # Calibration data
│   ├── 01/
│   ├── 02/
│   └── ...
├── poses/      # Ground truth poses
│   ├── 00.txt
│   ├── 01.txt 
│   └── ...
├── results/    # Results directory (will be created if not present)
│   ├── 00/
│   │   ├── 00_1.txt
│   │   └── ...
│   ├── 01/
│   └── ...
```

## Configuration

Edit `config.yaml` to set your parameters:

- `basedir`: Path to your KITTI dataset
- `results_dir`: Directory to store results
- `sequence`: KITTI sequence to process
- `detector`: Feature detector type (SIFT, ORB, etc.)
- `matcher`: Feature matcher type (BF, FLANN)
- `real_time`: Enable/disable real-time visualization
- Other parameters for feature matching and pose estimation





## Output

- Trajectory data is saved in KITTI format (3x4 transformation matrices)
- 3D visualization of the estimated camera trajectory
- Console output showing processing progress

## License

This project is open-source and available under the MIT License.


