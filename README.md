# Structure-from-Motion-and-3D-reconstruction
Reconstruction of stereo images to 3D models using OpenCV and Python

Steps for doing 3D reconstruction
- Camera calibration
- Undistort images
- Feature matching
- Reproject points and build point clouds

Use the chessboard images to calibrate your camera using calibration.py which generates the extrinsic parameters of a camera (like focal length, rotational and transational vectors, camera matrix) and saves it as a Numpy file. 
Reconstruct.py fetches these camera parameters and uses Semi-global Block Matching (SGBM) algorithm to do 3D reconstruction.
