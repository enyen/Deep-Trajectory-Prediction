# Deep-Trajectory-Prediction
Walking trajectory prediction in ROS

Given Map and object poses

### Method 1
[Least Squares Projections](human_motions/src/leastsquare_predict.cpp) in 2D space

### Method 2
[Points-to-Points NN](nn_trajectory/src/nn_predict.py) 2D points to 2D points prediction

### Method 3
[Scans-to-Points NN](nn_trajectory/src/nn_predict_scan.py) Ray-casted scan points to 2D points prediction
