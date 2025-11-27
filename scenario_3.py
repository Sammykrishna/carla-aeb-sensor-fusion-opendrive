# main.py - CARLA AEB System with FIXED Physics-Based Braking Logic
import math
import random
import os
import sys
import time
from collections import deque
import numpy as np
import pygame
from ultralytics import YOLO
import carla

# ------------------- Config -------------------
W, H = 960, 540
FOV = 90.0
FIXED_DT = 0.05  # 20 FPS simulation

# LiDAR tuning
LIDAR_RANGE = 60.0
LIDAR_CHANNELS = 64
LIDAR_PPS = 60000
LIDAR_ROT_HZ = int(1.0 / FIXED_DT)

# AEB logic tuning - FIXED PHYSICS-BASED
AEB_ARM_SPEED = 0.5   # m/s; don't AEB when parked
MIN_FWD_X = 2.0       # CRITICAL: Filter self-hits and very close points
MAX_LATERAL = 2.5     # meters (slightly wider for better coverage)
MIN_HEIGHT = -0.5     # meters (FIXED: allow points below camera mount)
MAX_HEIGHT = 3.0      # meters (FIXED: higher ceiling)
REACTION_TIME = 0.5   # seconds (reduced for faster response)
MAX_DECEL = 8.0       # m/s²
BRAKE_RELEASE_DISTANCE = 8.0  # meters (increased)
BRAKE_RELEASE_SPEED = 1.0     # m/s

# AEB State Machine
AEB_STATE_INACTIVE = 0
AEB_STATE_WARNING = 1
AEB_STATE_BRAKING = 2

CONTROLS_TEXT = [
    "Controls:",
    "  Up/W : Accelerate",
    "  Down/S : Brake",
    "  Left/A : Steer Left",
    "  Right/D : Steer Right",
    "  R : Hold to Reverse (disables AEB)",
    "  Q / ESC : Quit",
]

RELEVANT_CLASSES = {0, 1, 2, 3, 5, 7}
CLASS_NAMES = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorbike",
    5: "Bus", 7: "Truck"
}

# ------------------- Global Flags -------------------
simulation_active = False
latest_lidar_points = None
frame_surface = None
emergency_brake = False
aeb_active = False
aeb_state = AEB_STATE_INACTIVE
warning_active = False
min_distance_to_obstacle = float('inf')
aeb_history = deque(maxlen=3)
camera_ready = False
camera_error = None
detections = []
debug_lidar_points = []  # NEW: For visualization
# For Scenario 3: CCRb timing and state
obstacle_cruise_speed = 8.33  # ~30 km/h in m/s
obstacle_braking_decel = 6.0  # m/s²
obstacle_state = "cruising"   # or "braking"
obstacle_brake_start_time = None
CCRb_trigger_time = 3.0       # seconds after simulation starts

# ------------------- Helpers -------------------
def get_carla_version(client):
    """Determine CARLA version being used"""
    try:
        client.get_server_version()
        return "0.10"
    except:
        return "0.9"

def safe_find(bp_lib, bp_id, carla_version):
    """Safely find blueprint with version compatibility"""
    try:
        return bp_lib.find(bp_id)
    except:
        try:
            if carla_version == "0.10":
                alt_id = bp_id.replace('.', '_')
                return bp_lib.find(alt_id)
        except:
            pass

        try:
            if carla_version == "0.10":
                for year in [2017, 2018, 2019, 2020, 2021]:
                    try:
                        parts = bp_id.split('.')
                        if parts[-1].isdigit():
                            base = '.'.join(parts[:-1])
                            alt_id = f"{base}_{year}"
                        else:
                            alt_id = f"{bp_id}_{year}"
                        return bp_lib.find(alt_id)
                    except:
                        pass
        except:
            pass

        return None

def pick_vehicle_bp(bp_lib, client):
    """Robust vehicle blueprint selection"""
    carla_version = get_carla_version(client)
    print(f"Detected CARLA version: {carla_version}")

    preferred_vehicles = [
        "vehicle.lincoln.mkz2017", "vehicle.lincoln.mkz_2017",
        "vehicle.audi.tt", "vehicle.ford.mustang",
        "vehicle.mercedes.coupe", "vehicle.mercedes.coupe_2020",
        "vehicle.tesla.model3"
    ]

    for vid in preferred_vehicles:
        bp = safe_find(bp_lib, vid, carla_version)
        if bp is not None:
            print(f"Found vehicle blueprint: {vid}")
            return bp

    print("No preferred vehicle found, trying pattern matching...")
    for pattern in ["lincoln", "audi", "ford", "mercedes", "tesla"]:
        for bp in bp_lib.filter("vehicle.*"):
            if pattern in bp.id.lower():
                if bp.has_attribute("number_of_wheels") and bp.get_attribute("number_of_wheels").as_int() == 4:
                    print(f"Selected pattern-matched vehicle: {bp.id}")
                    return bp

    candidates = [
        bp for bp in bp_lib.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels") and
        bp.get_attribute("number_of_wheels").as_int() == 4
    ]

    if candidates:
        print(f"Selected fallback vehicle: {candidates[0].id}")
        return random.choice(candidates)

    raise RuntimeError("No vehicle blueprints available in this CARLA build.")

def transform_points_to_camera(points, lidar, camera):
    """FIXED: Properly transform LiDAR points to camera coordinates"""
    cam_tf = camera.get_transform()
    lidar_tf = lidar.get_transform()

    # Get inverse camera transform and LiDAR transform
    cam_inv_mat = np.array(cam_tf.get_inverse_matrix()).reshape(4, 4)
    lidar_mat = np.array(lidar_tf.get_matrix()).reshape(4, 4)

    N = points.shape[0]
    points_hom = np.concatenate([points, np.ones((N, 1))], axis=1)

    # LiDAR local -> World -> Camera local
    points_world = np.dot(points_hom, lidar_mat.T)
    points_cam = np.dot(points_world, cam_inv_mat.T)

    # Convert to Euclidean
    points_cam = points_cam[:, :3] / points_cam[:, 3][:, np.newaxis]

    # FIXED: Filter points in front AND apply MIN_FWD_X
    in_front = points_cam[:, 0] > MIN_FWD_X  # Changed from 0.1 to MIN_FWD_X
    points_cam = points_cam[in_front]

    return points_cam

def project_to_image(points_cam, fov, width, height):
    """Project camera coordinate points to 2D image coordinates"""
    if len(points_cam) == 0:
        return np.array([]), np.array([])

    focal_length = width / (2.0 * math.tan(fov * math.pi / 360.0))

    # Avoid division by zero
    valid_depth = points_cam[:, 0] > 0.1
    points_cam = points_cam[valid_depth]

    if len(points_cam) == 0:
        return np.array([]), np.array([])

    u = focal_length * (points_cam[:, 1] / points_cam[:, 0]) + (width / 2.0)
    v = focal_length * (points_cam[:, 2] / points_cam[:, 0]) + (height / 2.0)

    valid = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    return np.column_stack((u, v))[valid], points_cam[valid]

def create_straight_road_map():
    """Generate minimal OpenDRIVE file for straight road test"""
    opendrive_content = """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.opendrive.org/OpenDRIVE.xsd">
  <header revMajor="1" revMinor="4" name="StraightRoad" version="1.4" date="2024-04-01" north="0" south="0" east="0" west="0"/>
  <road name="StraightRoad" length="300.0" id="1" junction="-1">
    <link/>
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="300.0">
        <line/>
      </geometry>
    </planView>
    <elevationProfile>
      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
    </elevationProfile>
    <lanes>
      <laneSection s="0.0">
        <left>
          <lane id="1" type="none" level="false">
            <link/>
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark type="solid" weight="standard" color="white" width="0.1"/>
          </lane>
        </left>
        <center>
          <lane id="0" type="none" level="false">
            <link/>
            <width sOffset="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
            <roadMark type="solid" weight="standard" color="white" width="0.1"/>
          </lane>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link/>
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark type="solid" weight="standard" color="white" width="0.1"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
    <type s="0.0" type="town" roadType="11"/>
    <surface>
      <material s="0.0" type="0" friction="0.7" rollingResistance="0.01"/>
    </surface>
  </road>
</OpenDRIVE>"""
    return opendrive_content

def calculate_stopping_distance(current_speed, decel_rate, reaction_time):
    """Calculate stopping distance with vehicle dynamics model"""
    reaction_dist = current_speed * reaction_time
    braking_dist = (current_speed ** 2) / (2 * decel_rate)
    return (reaction_dist + braking_dist) * 1.15  # 15% safety margin

def calculate_ttc(distance, speed):
    """Calculate Time-to-Collision"""
    if speed <= 0.1:
        return float('inf')
    return distance / speed

# ------------------- Sensor Callbacks -------------------
def lidar_callback(pc):
    global latest_lidar_points
    if not simulation_active:
        return

    try:
        pts = np.frombuffer(pc.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        latest_lidar_points = pts
    except Exception as e:
        print(f"⚠️ LiDAR callback error: {e}")

def camera_callback(image, model, vehicle, camera, lidar):
    global emergency_brake, frame_surface, latest_lidar_points, aeb_active, aeb_state, warning_active
    global min_distance_to_obstacle, aeb_history, camera_ready, camera_error, detections, debug_lidar_points

    if not simulation_active:
        return

    try:
        camera_ready = True
        detections = []
        debug_lidar_points = []

        # Process image
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, :3]

        # Run YOLO with lower confidence for better detection
        results = model.predict(img, conf=0.25, verbose=False)

        # Get vehicle speed
        v = vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_kmh = speed * 3.6

        min_distance = float('inf')
        ttc = float('inf')
        high_risk = False
        stopping_dist = float('inf')
        has_valid_data = False

        # NEW: Pure LiDAR fallback detection
        lidar_only_min_dist = float('inf')

        if speed > AEB_ARM_SPEED and latest_lidar_points is not None and len(latest_lidar_points) > 0:
            # Transform LiDAR points
            points_cam = transform_points_to_camera(latest_lidar_points, lidar, camera)

            if len(points_cam) > 0:
                # Project to image
                points_2d, points_cam = project_to_image(points_cam, FOV, W, H)

                if len(points_2d) > 0:
                    # FIXED: Height and lateral filtering
                    lateral = np.abs(points_cam[:, 1])
                    height = points_cam[:, 2]

                    valid_points = (
                        (lateral <= MAX_LATERAL) &
                        (height >= MIN_HEIGHT) &
                        (height <= MAX_HEIGHT)
                    )

                    filtered_2d = points_2d[valid_points]
                    filtered_cam = points_cam[valid_points]

                    # Store for visualization
                    debug_lidar_points = list(zip(filtered_2d, filtered_cam))

                    if len(filtered_cam) > 0:
                        has_valid_data = True

                        # NEW: Calculate pure LiDAR minimum distance (FALLBACK)
                        forward_distances = filtered_cam[:, 0]  # X coordinate = forward distance
                        lidar_only_min_dist = np.min(forward_distances)

                        print(f"[LiDAR] Detected {len(filtered_cam)} points in valid zone, closest: {lidar_only_min_dist:.2f}m")

                        # Process YOLO detections with LiDAR fusion
                        for r in results:
                            for box in getattr(r, "boxes", []):
                                try:
                                    cls_id = int(box.cls[0].item())
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                                    if cls_id in RELEVANT_CLASSES:
                                        # Find points in this box
                                        in_box = (
                                            (filtered_2d[:, 0] >= x1) & (filtered_2d[:, 0] <= x2) &
                                            (filtered_2d[:, 1] >= y1) & (filtered_2d[:, 1] <= y2)
                                        )

                                        box_points = filtered_cam[in_box]

                                        if len(box_points) > 0:
                                            distances = box_points[:, 0]
                                            min_dist = np.min(distances)
                                            min_distance = min(min_distance, min_dist)

                                            ttc = calculate_ttc(min_dist, speed)
                                            stopping_dist = calculate_stopping_distance(speed, MAX_DECEL, REACTION_TIME)

                                            # FIXED: Confidence calculation
                                            # 1. LiDAR confidence (FIXED: Lower threshold for close objects)
                                            lidar_point_threshold = max(3, int(10 - min_dist * 0.5))  # Adaptive threshold
                                            lidar_confidence = min(1.0, len(box_points) / lidar_point_threshold)

                                            # 2. YOLO confidence (FIXED: Extract actual confidence)
                                            camera_confidence = 0.5  # Default
                                            if hasattr(box, 'conf') and len(box.conf) > 0:
                                                camera_confidence = float(box.conf[0].item())

                                            # 3. FIXED: Distance-based weighting (LiDAR better at close range!)
                                            # At 0m: 100% LiDAR, 0% camera
                                            # At 20m+: 30% LiDAR, 70% camera
                                            camera_weight = min(0.7, max(0.0, min_dist / 30.0))
                                            lidar_weight = 1.0 - camera_weight

                                            # Final confidence
                                            detection_confidence = (
                                                lidar_confidence * lidar_weight +
                                                camera_confidence * camera_weight
                                            )

                                            # BONUS: Boost confidence if distance is very close
                                            if min_dist < 15.0:
                                                detection_confidence = min(1.0, detection_confidence * 1.3)

                                            print(f"[FUSION] {CLASS_NAMES.get(cls_id, 'object')} at {min_dist:.2f}m | "
                                                  f"LiDAR: {len(box_points)}pts ({lidar_confidence:.2f}) | "
                                                  f"YOLO: {camera_confidence:.2f} | "
                                                  f"Final: {detection_confidence:.2f} | TTC: {ttc:.2f}s")

                                            # FIXED: Risk assessment
                                            confidence_threshold = 0.15  # Lower threshold
                                            min_lidar_points = 2  # Require at least 2 points

                                            # Trigger conditions:
                                            # 1. TTC < 2.0 seconds OR distance < stopping distance
                                            # 2. Confidence > threshold
                                            # 3. Minimum LiDAR points
                                            if ((ttc < 2.0 or min_dist < stopping_dist * 1.1) and
                                                detection_confidence > confidence_threshold and
                                                len(box_points) >= min_lidar_points):
                                                high_risk = True
                                                print(f"  >>> FUSION HIGH RISK: TTC={ttc:.2f}s, Dist={min_dist:.2f}m, Conf={detection_confidence:.2f}")

                                            detections.append({
                                                'cls_id': cls_id,
                                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                                'confidence': detection_confidence,
                                                'distance': min_dist,
                                                'ttc': ttc,
                                                'lidar_points': len(box_points)
                                            })
                                except Exception as e:
                                    print(f"Error processing detection: {e}")
                                    continue

                        # NEW: FALLBACK - Pure LiDAR Detection (if YOLO missed it)
                        if not high_risk and lidar_only_min_dist < 20.0:
                            stopping_dist = calculate_stopping_distance(speed, MAX_DECEL, REACTION_TIME)
                            ttc_lidar = calculate_ttc(lidar_only_min_dist, speed)

                            # Aggressive LiDAR-only trigger for very close objects
                            if lidar_only_min_dist < stopping_dist * 0.8 or ttc_lidar < 1.5:
                                high_risk = True
                                min_distance = lidar_only_min_dist
                                print(f"  >>> LIDAR-ONLY FALLBACK TRIGGER: Dist={lidar_only_min_dist:.2f}m, TTC={ttc_lidar:.2f}s")

                                # Add visual indicator for LiDAR-only detection
                                detections.append({
                                    'cls_id': -1,  # Special marker for LiDAR-only
                                    'x1': W//2 - 50, 'y1': H//2 - 50,
                                    'x2': W//2 + 50, 'y2': H//2 + 50,
                                    'confidence': 0.9,
                                    'distance': lidar_only_min_dist,
                                    'ttc': ttc_lidar,
                                    'lidar_points': len(filtered_cam)
                                })

        # Update tracking
        if min_distance < float('inf'):
            min_distance_to_obstacle = min_distance
        else:
            min_distance_to_obstacle = lidar_only_min_dist

        # Hysteresis
        aeb_history.append(high_risk)
        should_brake = sum(aeb_history) >= 2

        # State machine
        if should_brake and has_valid_data:
            if aeb_state == AEB_STATE_INACTIVE:
                aeb_state = AEB_STATE_WARNING
                warning_active = True
                print("⚠️ AEB WARNING: Potential collision")
            elif aeb_state == AEB_STATE_WARNING:
                aeb_state = AEB_STATE_BRAKING
                print("🛑 AEB ACTIVATED: Emergency braking")
        else:
            if aeb_state == AEB_STATE_WARNING:
                aeb_state = AEB_STATE_INACTIVE
                warning_active = False
            elif aeb_state == AEB_STATE_BRAKING:
                if min_distance_to_obstacle > BRAKE_RELEASE_DISTANCE and speed < BRAKE_RELEASE_SPEED:
                    aeb_state = AEB_STATE_INACTIVE
                    print("✅ AEB RELEASED")

        emergency_brake = (aeb_state == AEB_STATE_BRAKING)
        aeb_active = (aeb_state != AEB_STATE_INACTIVE)

        # Render
        frame_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

        # Draw LiDAR points (NEW: Debug visualization)
        for pt_2d, pt_3d in debug_lidar_points:
            x, y = int(pt_2d[0]), int(pt_2d[1])
            distance = pt_3d[0]
            # Color based on distance: red=close, green=far
            color_val = int(255 * min(1.0, distance / 30.0))
            pygame.draw.circle(frame_surface, (255 - color_val, color_val, 0), (x, y), 2)

        # Draw detections
        for det in detections:
            cls_id = det['cls_id']
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            confidence = det['confidence']
            distance = det['distance']
            ttc = det['ttc']
            lidar_pts = det.get('lidar_points', 0)

            if cls_id == -1:  # LiDAR-only detection
                color = (255, 255, 0)  # Yellow
                label = "LiDAR-ONLY"
            elif cls_id in RELEVANT_CLASSES:
                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                label = CLASS_NAMES.get(cls_id, 'Object')
            else:
                continue

            # Draw box
            pygame.draw.rect(frame_surface, color, (x1, y1, x2 - x1, y2 - y1), 3)

            # Draw label
            font = pygame.font.SysFont(None, 24)
            text = font.render(f"{label} ({confidence:.2f})", True, color)
            frame_surface.blit(text, (x1, y1 - 20))

            # Draw distance/TTC
            info_text = font.render(f"{distance:.1f}m | {lidar_pts}pts | TTC:{ttc:.1f}s", True, color)
            frame_surface.blit(info_text, (x1, y1 - 45))

        # Status overlay
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Speed: {speed_kmh:.1f} km/h", True, (255, 255, 255))
        frame_surface.blit(speed_text, (10, H - 50))

        if min_distance_to_obstacle < float('inf'):
            dist_text = font.render(f"Closest: {min_distance_to_obstacle:.1f}m", True, (255, 255, 255))
            frame_surface.blit(dist_text, (10, H - 25))

        if aeb_state == AEB_STATE_BRAKING:
            warn = font.render("🛑 EMERGENCY BRAKING ACTIVE 🛑", True, (255, 0, 0))
            frame_surface.blit(warn, (W // 2 - warn.get_width() // 2, 10))
        elif aeb_state == AEB_STATE_WARNING:
            warn = font.render("⚠️ COLLISION WARNING ⚠️", True, (255, 255, 0))
            frame_surface.blit(warn, (W // 2 - warn.get_width() // 2, 10))

    except Exception as e:
        camera_error = e
        print(f"❌ Camera callback error: {e}")
        import traceback
        traceback.print_exc()

# ------------------- Main -------------------
def main():
    global simulation_active, obstacle_state, obstacle_brake_start_time

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CARLA AEB – FIXED Version")
    clock = pygame.time.Clock()

    print("="*50)
    print("CARLA AEB System - FIXED VERSION")
    print("="*50)

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    print("⏳ Waiting for CARLA server...")
    time.sleep(5)

    vehicle = None
    camera = None
    lidar = None
    obstacle = None
    world = None
    original_settings = None

    try:
        try:
            server_version = client.get_server_version()
            print(f"CARLA Server Version: {server_version}")
        except:
            print("CARLA Server Version: Unknown")

        print("\nLoading OpenDRIVE map...")
        opendrive_content = create_straight_road_map()

        world = client.generate_opendrive_world(
            opendrive_content,
            carla.OpendriveGenerationParameters(
                vertex_distance=2.0,
                max_road_length=100.0,
                wall_height=1.0,
                additional_width=0.6,
                smooth_junctions=True,
                enable_mesh_visibility=True
            )
        )
        print("✅ OpenDRIVE map loaded")

        settings = world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)
        print("✅ Synchronous mode enabled")

        world.set_weather(carla.WeatherParameters.ClearNoon)
        print("✅ Weather set")

        carla_map = world.get_map()
        map_spawn_points = carla_map.get_spawn_points()
        print(f"Found {len(map_spawn_points)} spawn points")

        if len(map_spawn_points) == 0:
            print("⚠️ No spawn points - using fallback")
            map_spawn_points = [carla.Transform(carla.Location(x=5.0, y=0.0, z=0.2), carla.Rotation(yaw=0.0))]

        bp_lib = world.get_blueprint_library()
        print("\nSpawning ego vehicle...")
        vehicle_bp = pick_vehicle_bp(bp_lib, client)
        print(f"Selected vehicle: {vehicle_bp.id}")

        vehicle = None
        for i, sp in enumerate(map_spawn_points):
            try:
                vehicle = world.try_spawn_actor(vehicle_bp, sp)
                if vehicle:
                    print(f"✅ Ego vehicle spawned at {sp.location}")
                    break
            except Exception as e:
                print(f"❌ Spawn attempt {i+1} failed: {e}")

        if vehicle is None:
            spawn_location = carla.Location(x=5.0, y=0.0, z=0.2)
            vehicle = world.spawn_actor(vehicle_bp, carla.Transform(spawn_location, carla.Rotation(yaw=0.0)))
            print(f"✅ Fallback spawn successful")

        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        # Spawn obstacle
        print("\nSpawning obstacle...")
        tesla_names = ['vehicle.tesla.model3', 'vehicle.tesla.model3_2017', 'vehicle.tesla.model3_2020']
        obstacle_bp = None
        for name in tesla_names:
            try:
                obstacle_bp = bp_lib.find(name)
                if obstacle_bp:
                    print(f"✅ Found obstacle: {name}")
                    break
            except:
                continue

        if obstacle_bp is None:
            vehicle_blueprints = bp_lib.filter('vehicle.*')
            valid_vehicles = [
                bp for bp in vehicle_blueprints
                if bp.has_attribute('number_of_wheels') and
                bp.get_attribute('number_of_wheels').as_int() == 4
            ]
            if valid_vehicles:
                obstacle_bp = random.choice(valid_vehicles)
                print(f"✅ Fallback obstacle: {obstacle_bp.id}")
            else:
                raise RuntimeError("No obstacle blueprint available")

        obstacle_transform = carla.Transform(carla.Location(x=30.0, y=0.0, z=0.2), carla.Rotation(yaw=0.0))
        obstacle = world.spawn_actor(obstacle_bp, obstacle_transform)
        obstacle.set_simulate_physics(True)
        print("✅ Obstacle spawned at 50m (CCRb: will cruise then brake)")


        # Setup sensors
        print("\nSetting up sensors...")
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(W))
        cam_bp.set_attribute("image_size_y", str(H))
        cam_bp.set_attribute("fov", str(FOV))
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))
        cam_bp.set_attribute("gamma", "2.2")

        front_offset = vehicle.bounding_box.extent.x + 1.50
        cam_tf = carla.Transform(
            carla.Location(x=front_offset, y=0.0, z=1.8),
            carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0)
        )
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        print("✅ Camera created")

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("points_per_second", str(LIDAR_PPS))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROT_HZ))
        lidar_bp.set_attribute("sensor_tick", str(FIXED_DT))
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-30.0")
        lidar_tf = carla.Transform(carla.Location(z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        print("✅ LiDAR created")

        # Load YOLO
        print("\nLoading YOLO model...")
        try:
            model = YOLO("yolov8n.pt")
            print("✅ YOLOv8n loaded")
        except Exception as e:
            print(f"❌ YOLO error: {e}")
            sys.exit(1)

        # Start sensors
        camera.listen(lambda img: camera_callback(img, model, vehicle, camera, lidar))
        print("✅ Camera listening")
        lidar.listen(lidar_callback)
        print("✅ LiDAR listening")

        print("\n=== SYSTEM READY ===")
        print("Drive with WASD. AEB will trigger automatically.")
        print("====================\n")

        simulation_active = True
        running = True

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            keys = pygame.key.get_pressed()
            steer = 0.0
            throttle = 0.0
            brake = 0.0
            reverse = False

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer = -0.5
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steer = 0.5
            if keys[pygame.K_r]:
                reverse = True
                global aeb_active, emergency_brake, aeb_state
                aeb_state = AEB_STATE_INACTIVE
                aeb_active = False
                emergency_brake = False
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle = 0.5
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                brake = 1.0

            # AEB override
            if aeb_state == AEB_STATE_BRAKING:
                v = vehicle.get_velocity()
                speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                ttc = min_distance_to_obstacle / speed if speed > 0.1 else float('inf')

                # Progressive braking
                brake_force = min(1.0, max(0.3, 2.5 - 1.5 * ttc))
                ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake_force, reverse=False)
                print(f"BRAKE FORCE: {brake_force:.2f} (TTC={ttc:.2f}s, Dist={min_distance_to_obstacle:.2f}m)")
            else:
                ctrl = carla.VehicleControl(
                    throttle=max(0.0, min(1.0, throttle)),
                    steer=max(-1.0, min(1.0, steer)),
                    brake=max(0.0, min(1.0, brake)),
                    reverse=reverse
                )

            try:
                vehicle.apply_control(ctrl)
            except Exception as e:
                print(f"Control error: {e}")
                running = False

            # --- Scenario 3: CCRb Obstacle Behavior ---
            # --- Scenario 3: CCRb Obstacle Behavior (Distance-Based) ---
            if obstacle is not None and simulation_active:
                try:
                    obs_loc = obstacle.get_location()
                    # Start braking after traveling past x = 60.0 (i.e., ~20m from spawn at x=40)
                    if obstacle_state == "cruising":
                        obstacle.apply_control(carla.VehicleControl(throttle=0.4, brake=0.0, steer=0.0))
                        if obs_loc.x >= 60.0:
                            obstacle_state = "braking"
                            print("⚠️ Obstacle started hard braking (CCRb)")
                    elif obstacle_state == "braking":
                        obstacle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0))
                except Exception as e:
                    print(f"Obstacle control error: {e}")

            try:
                world.tick()
            except Exception as e:
                print(f"Tick error: {e}")
                running = False

            if frame_surface is not None:
                screen.blit(frame_surface, (0, 0))

                font = pygame.font.SysFont(None, 24)
                y = 10
                for line in CONTROLS_TEXT:
                    screen.blit(font.render(line, True, (255, 255, 255)), (10, y))
                    y += 22

                pygame.display.flip()
            else:
                screen.fill((0, 0, 0))
                font = pygame.font.SysFont(None, 36)
                error_msg = font.render("Waiting for camera... Press Q to quit", True, (255, 255, 255))
                screen.blit(error_msg, (W // 2 - error_msg.get_width() // 2, H // 2))
                pygame.display.flip()

            clock.tick(60)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        simulation_active = False
        time.sleep(1.0)

        try:
            if camera is not None:
                camera.stop()
                print("✅ Camera stopped")
            if lidar is not None:
                lidar.stop()
                print("✅ LiDAR stopped")
            if world is not None and original_settings is not None:
                world.apply_settings(original_settings)
                print("✅ Settings restored")
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

        actors = [camera, lidar, vehicle, obstacle]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                    print(f"✅ Destroyed {actor.type_id}")
                except Exception as e:
                    print(f"❌ Destroy error: {e}")

        pygame.quit()
        print("Cleanup complete.")

if __name__ == "__main__":
    print("Starting FIXED CARLA AEB System...")
    main()
    print("Program terminated.")
