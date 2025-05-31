# Tesla AI Day 2021
## Computer Vision
- There are 8 cameras positioned around the vehicle
- It must process data in real time: 3D positions of lines, edges, curbs, traffic signs, traffic lights, car's orientation, position, depth, velocities etc.
- 8 Cameras -> 3D "Vector Space"
- Cars are synthetic animals and senses everything. Here, they are building from scratch
- Images of 1280 x 960 12-Bit (HDR) @ 36Hz. They feed it to a Neural Network (Residual NN)
- RegNets give out images of varities: high spatial resolution with low channel count to low spatial resolution with high channel count
- Multi-Scale Feature Pyramic Fusion
- Detection Head: For each pixel: Is there an object here? If there were an object here, what is its extend and attributes?
- Multi-Task Learning "HydraNets": Object Detection, Traffic Lights Task, Lane Prediction etc.
### Occupancy Tracker (Stitching Up Individual Per-Camera Road Edge Predictions Across Cameras & Time)
- Problem 1: The across-camera fusion and the tracker are very difficult to write explicitly
- Problem 2: Image space is not the right output space
- Developed using C++
- Per pixel, you need to have more accurate depth prediction
### Multi-Cam Vector Space Predictions
- Problem 1: How do you transform features from image space to vector space?
- Problem 2: Vector space predictions require vector space datasets
