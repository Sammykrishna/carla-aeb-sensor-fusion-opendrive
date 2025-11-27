# carla-aeb-sensor-fusion-opendrive
Sensor fusion-based AEB in CARLA 0.10 with custom OpenDRIVE map and YOLOv8+LiDAR fusion
# Sensor Fusion-Based Automatic Emergency Braking in CARLA 0.10

This repository contains the code and report for my scientific project on a
sensor fusion-based Automatic Emergency Braking (AEB) system in CARLA 0.10.
The system combines YOLOv8 camera detections with LiDAR point clouds and runs
on a **custom OpenDRIVE map that I generated programmatically in Python**.

## Key Features

- **Custom OpenDRIVE test track**  
  I implemented `create_straight_road_map()` which builds a minimal 300 m
  straight road directly as OpenDRIVE XML inside the Python script.  
  This map is then loaded via `client.generate_opendrive_world(...)`, so no
  external `.xodr` file is needed. All lane widths, markings, and surface
  parameters are explicitly defined in code by me. 

- **Physics-based AEB logic (written by me)**  
  I designed the AEB decision logic using:
  - A kinematic stopping distance model (`calculate_stopping_distance`)
  - Time-to-collision (`calculate_ttc`)
  - A three-state AEB state machine (INACTIVE → WARNING → BRAKING) with
    hysteresis to avoid flickering.
  - Progressive brake force based on TTC and distance.

- **Sensor fusion: YOLOv8 + LiDAR**
  - YOLOv8 detects relevant classes (car, truck, bus, pedestrian, etc.)
  - LiDAR points are transformed into the camera frame and projected into
    image coordinates.
  - For each detection, I fuse camera confidence with LiDAR point density and
    distance to compute a final risk score and trigger AEB when thresholds
    are exceeded.

- **Euro NCAP-inspired scenarios**
  - Scenario 1 – CCRs (stationary target)
  - Scenario 2 – CCRm (moving target with constant speed)
  - Scenario 3 – CCRb (cut-in / braking leading vehicle)

Each scenario is implemented as a separate Python script with the same
underlying AEB and OpenDRIVE infrastructure.

