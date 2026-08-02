# Sensor Fusion-Based Automatic Emergency Braking in CARLA 0.10

A sensor fusion AEB pipeline built in CARLA 0.10, running on custom OpenDRIVE road networks that I designed and generated programmatically in Python. The system fuses YOLOv8 camera detections with 64-channel LiDAR point clouds to estimate time-to-collision and trigger physics-based emergency braking, and it was evaluated across three Euro NCAP-inspired scenarios.

This project was submitted as a Scientific Project for the M.Sc. Mechatronics program at RWU Weingarten, together with Meghana Ratnam Gudimetla. The full write-up, with methodology, calibration steps, and results, is in [`Scientific_Project.pdf`](Scientific_Project.pdf).

## Demo videos

| Scenario | What happens | Video |
|---|---|---|
| 1: CCRs | Ego car approaches a stationary lead car | *add video link* |
| 2: CCRm | Ego car approaches a lead car moving at a constant 20 km/h | *add video link* |
| 3: CCRb | Lead car drives ahead then brakes hard | *add video link* |

## What the system does

- Runs CARLA in synchronous mode at a fixed 20 Hz tick (`Δt = 0.05 s`) so every run is deterministic and repeatable.
- Detects objects with YOLOv8n from a forward-facing RGB camera, and projects LiDAR points into the camera frame to get range and point density per detection.
- Fuses camera confidence with LiDAR distance and density into a single risk score, giving more weight to LiDAR at close range and falling back to LiDAR-only braking when the camera misses a target.
- Drives a three-state AEB state machine (INACTIVE → WARNING → BRAKING) using a kinematic stopping distance model and time-to-collision, with majority-vote hysteresis so the state doesn't flicker.
- Shapes the brake command progressively based on TTC and distance rather than slamming to full brake immediately.
- Overlays live detections, LiDAR density, closest object, and TTC on a Pygame window while it runs.

## How I built the OpenDRIVE maps

I wanted repeatable, lightweight test tracks instead of loading CARLA's full Town maps, which are asset-heavy and eat into the frame budget I needed for YOLO and LiDAR processing. So instead of using a pre-built map, I generate the OpenDRIVE XML directly in Python and hand it to CARLA at runtime through `client.generate_opendrive_world(...)`. No `.xodr` file on disk, no Town assets, just the road geometry the scenario actually needs.

I implemented `create_straight_road_map()`, which builds a 300 m single-carriageway straight road directly as OpenDRIVE XML inside the Python script. Lane widths, road markings, and elevation profile are all defined explicitly in the XML, so I can change the test track just by editing numbers in the string, no map editor needed.

Here's a shortened example of what the map looks like as raw OpenDRIVE:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.opendrive.org/OpenDRIVE.xsd">
  <header revMajor="1" revMinor="4" name="StraightRoad" version="1.4" date="2024-04-01" north="0" south="0" east="0" west="0"/>
  <road name="StraightRoad" length="300.0" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="300.0">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <left>
          <lane id="1" type="none" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark type="solid" weight="standard" color="white" width="0.1"/>
          </lane>
        </left>
        ...
```

This approach kept the environment sparse enough to hold a steady 20 Hz simulation rate while YOLO, LiDAR fusion, and the AEB logic run every tick.

## Sensor suite and fusion

- **Camera**: forward RGB, used for YOLOv8n class detection (car, truck, bus, pedestrian, etc.).
- **LiDAR**: roof-mounted, 64 channels, 60 m range, 60,000 points/sec, rotating at the same 20 Hz as the simulation tick.
- **Calibration**: camera intrinsics, LiDAR channel elevation angles, and the LiDAR-to-camera extrinsic transform are all derived from CARLA's own actor transforms, so the projection from LiDAR points to image pixels stays consistent across runs.
- **Fusion**: for every YOLO detection, I project nearby LiDAR points into the image, gate them to a forward corridor and height band, and combine LiDAR point density with camera confidence into one risk score per object. If the camera fails to pick up a target, a LiDAR-only safety path can still trigger braking at close range.

## AEB decision logic

- **Stopping distance model**: kinematic model using reaction time and maximum deceleration.
- **Time-to-collision**: computed per object per frame from range and closing speed.
- **State machine**: INACTIVE → WARNING → BRAKING, with majority voting across recent frames to avoid state flicker from a single noisy detection.
- **Brake shaping**: brake torque scales with how bad the TTC and distance look, rather than binary on/off braking.

## Scenarios tested

All scenarios are Euro NCAP-inspired car-to-car and car-to-pedestrian AEB tests, each run multiple times across a speed sweep.

| Scenario | Full name | Setup | Result |
|---|---|---|---|
| 1: CCRs | Car-to-Car Rear Stationary | Ego approaches a stationary target, 10–50 km/h sweep | 19/20 runs avoided collision, up to ~39.7 km/h. Failed at ~49 km/h (camera missed the stationary target at range, LiDAR fallback triggered too late) |
| 2: CCRm | Car-to-Car Rear Moving | Lead car at a constant ~22 km/h, ego 31–59 km/h | 20/20 runs avoided collision, minimum TTC never dropped below ~1.08 s |
| 3: CCRb | Car-to-Car Rear Braking | Lead car at 30 km/h brakes at ~-6 m/s², ego 31–60 km/h | 20/20 runs avoided collision, even at the highest closing speed tested |

All three scenarios are covered in full detail, with speed sweeps, TTC plots, and braking jerk analysis, in the report. The main limitation found there is stationary-target detection at long range and high closing speed, which is discussed in the report's future work section.

## Repository structure

```
.
├── scenario_1.py          # CCRs: stationary target
├── scenario_2.py          # CCRm: moving target
├── scenario_3.py          # CCRb: braking lead vehicle
├── Scientific_Project.pdf # Full report
└── README.md
```

## Setup

Tested on CARLA 0.10.0, Windows 11, Python 3.10.

```bash
# create an isolated environment
py -3.10 -m venv carla-aeb
carla-aeb\Scripts\activate

# install dependencies
pip install --upgrade pip
pip install pygame numpy ultralytics

# install the CARLA client wheel from the CARLA package
cd <path-to-carla>\PythonAPI\dist
pip install carla-0.10.0-*-win_amd64.whl
```

Start the CARLA server, then run any scenario directly:

```bash
python scenario_1.py
```

## Report

The full report covers CARLA setup, sensor calibration derivations, the fusion and AEB math, and the complete Euro NCAP results with plots and tables: [`Scientific_Project.pdf`](Scientific_Project.pdf).

## Future work

From the report's conclusions:

- Improve stationary-object detection at long range and high closing speed (better small-object detection, temporal confidence aggregation, stronger LiDAR weighting beyond 30 m).
- Add a radar stream to help separate stationary objects from ego motion.
- Move to staged, jerk-limited braking instead of a single braking curve, to reduce peak jerk without increasing stopping distance.
- Extend the OpenDRIVE maps beyond a straight road, to curved roads and intersections.

## Authors

- Samanth Krishna ([github.com/Sammykrishna](https://github.com/Sammykrishna))
- Meghana Ratnam Gudimetla

Guided by Prof. Dr. Stefan Elser, RWU Ravensburg-Weingarten.
