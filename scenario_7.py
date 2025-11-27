# scenario_7_fixed.py - CARLA AEB Scenario 7: Car-to-Pedestrian Nearside Adult (CPNA)
# FIXED: Uses WalkerControl API instead of AI navigation or set_target_velocity
# WalkerControl must be applied every frame for continuous movement
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
FIXED_DT = 0.05

# LiDAR tuning - ENHANCED for long-range detection
LIDAR_RANGE = 200.0  # Increased from 60m to 200m (full road coverage)
LIDAR_CHANNELS = 128  # Increased from 64 to 128 (better resolution)
LIDAR_PPS = 120000  # Increased from 60000 to 120000 (2x points per second)
LIDAR_ROT_HZ = int(1.0 / FIXED_DT)

# AEB logic - INCLUDING PEDESTRIANS - ENHANCED SENSITIVITY
AEB_ARM_SPEED = 0.3  # Lowered from 0.5 (activate at lower speed)
MIN_FWD_X = 2.0
MAX_LATERAL = 3.5  # Increased from 2.5 (wider detection zone for pedestrians)
MIN_HEIGHT = -0.5  # Lowered from -0.3
MAX_HEIGHT = 2.8  # Increased from 2.5 (better pedestrian height coverage)
REACTION_TIME = 0.5
MAX_DECEL = 9.0  # Increased from 8.0 (more aggressive braking)
BRAKE_RELEASE_DISTANCE = 10.0  # Increased from 8.0
BRAKE_RELEASE_SPEED = 1.0

# Pedestrian trigger settings
PEDESTRIAN_TRIGGER_DISTANCE = 100.0  # meters - when to start crossing
PEDESTRIAN_WALKING_SPEED = 2.5  # m/s (~8 km/h - fast walk/slow jog)

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
    "  R : Reverse (disables AEB)",
    "  P : Manually trigger pedestrian crossing",
    "  Q/ESC : Quit",
]

RELEVANT_CLASSES = {0, 1, 2, 3, 5, 7}  # Including 0 for pedestrians
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
debug_lidar_points = []
pedestrian_crossing = False
pedestrian_triggered = False

# ------------------- Helpers -------------------
def get_carla_version(client):
    try:
        client.get_server_version()
        return "0.10"
    except:
        return "0.9"

def safe_find(bp_lib, bp_id, carla_version):
    try:
        return bp_lib.find(bp_id)
    except:
        if carla_version == "0.10":
            try:
                return bp_lib.find(bp_id.replace('.', '_'))
            except:
                pass
        return None

def pick_vehicle_bp(bp_lib, client):
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

    candidates = [
        bp for bp in bp_lib.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels") and
        bp.get_attribute("number_of_wheels").as_int() == 4
    ]

    if candidates:
        return random.choice(candidates)

    raise RuntimeError("No vehicle blueprints available")

def create_pedestrian_road_map():
    """OpenDRIVE map for Scenario 7: Pedestrian Crossing"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.opendrive.org/OpenDRIVE.xsd">
  <header revMajor="1" revMinor="4" name="CPNA_StraightRoad" version="1.4" date="2025-10-17" north="0" south="0" east="0" west="0"/>
  <road name="PedCrossing" length="200.0" id="1" junction="-1">
    <link/>
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="200.0">
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

def transform_points_to_camera(points, lidar, camera):
    cam_tf = camera.get_transform()
    lidar_tf = lidar.get_transform()
    cam_inv_mat = np.array(cam_tf.get_inverse_matrix()).reshape(4, 4)
    lidar_mat = np.array(lidar_tf.get_matrix()).reshape(4, 4)
    N = points.shape[0]
    points_hom = np.concatenate([points, np.ones((N, 1))], axis=1)
    points_world = np.dot(points_hom, lidar_mat.T)
    points_cam = np.dot(points_world, cam_inv_mat.T)
    points_cam = points_cam[:, :3] / points_cam[:, 3][:, np.newaxis]
    in_front = points_cam[:, 0] > MIN_FWD_X
    return points_cam[in_front]

def project_to_image(points_cam, fov, width, height):
    if len(points_cam) == 0:
        return np.array([]), np.array([])
    focal_length = width / (2.0 * math.tan(fov * math.pi / 360.0))
    valid_depth = points_cam[:, 0] > 0.1
    points_cam = points_cam[valid_depth]
    if len(points_cam) == 0:
        return np.array([]), np.array([])
    u = focal_length * (points_cam[:, 1] / points_cam[:, 0]) + (width / 2.0)
    v = focal_length * (points_cam[:, 2] / points_cam[:, 0]) + (height / 2.0)
    valid = ((u >= 0) & (u < width) & (v >= 0) & (v < height))
    return np.column_stack((u, v))[valid], points_cam[valid]

def calculate_stopping_distance(current_speed, decel_rate, reaction_time):
    reaction_dist = current_speed * reaction_time
    braking_dist = (current_speed ** 2) / (2 * decel_rate)
    return (reaction_dist + braking_dist) * 1.15

def calculate_ttc(distance, speed):
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
        print(f"⚠️ LiDAR error: {e}")

def camera_callback(image, model, vehicle, camera, lidar):
    global emergency_brake, frame_surface, latest_lidar_points, aeb_active, aeb_state, warning_active
    global min_distance_to_obstacle, aeb_history, camera_ready, detections, debug_lidar_points

    if not simulation_active:
        return

    try:
        camera_ready = True
        detections = []
        debug_lidar_points = []

        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, :3]
        results = model.predict(img, conf=0.15, verbose=False)  # Very low threshold for maximum sensitivity

        v = vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_kmh = speed * 3.6

        min_distance = float('inf')
        high_risk = False
        has_valid_detection = False
        pedestrian_detected = False

        if speed > AEB_ARM_SPEED and latest_lidar_points is not None and len(latest_lidar_points) > 0:
            points_cam = transform_points_to_camera(latest_lidar_points, lidar, camera)
            if len(points_cam) > 0:
                points_2d, points_cam = project_to_image(points_cam, FOV, W, H)
                if len(points_2d) > 0:
                    lateral = np.abs(points_cam[:, 1])
                    height = points_cam[:, 2]
                    valid_points = (
                        (lateral <= MAX_LATERAL) &
                        (height >= MIN_HEIGHT) &
                        (height <= MAX_HEIGHT)
                    )
                    filtered_2d = points_2d[valid_points]
                    filtered_cam = points_cam[valid_points]
                    debug_lidar_points = list(zip(filtered_2d, filtered_cam))

                    if len(filtered_cam) > 0:
                        for r in results:
                            for box in getattr(r, "boxes", []):
                                try:
                                    cls_id = int(box.cls[0].item())

                                    # Accept pedestrians (0), vehicles, and cyclists
                                    if cls_id not in RELEVANT_CLASSES:
                                        continue

                                    if cls_id == 0:
                                        pedestrian_detected = True

                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

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

                                        lidar_confidence = min(1.0, len(box_points) / 6.0)  # Lower divisor for easier confidence
                                        camera_confidence = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.5
                                        detection_confidence = (lidar_confidence * 0.6 + camera_confidence * 0.4)  # Weight LiDAR more

                                        # Much more sensitive for pedestrians - ENHANCED
                                        confidence_threshold = 0.15 if cls_id == 0 else 0.3  # Lowered significantly
                                        min_lidar_points = 2 if cls_id == 0 else 4  # Reduced minimum points

                                        # More aggressive triggering - increased TTC and distance thresholds
                                        if ((ttc < 4.5 or min_dist < stopping_dist * 1.8) and  # Increased from 3.0 and 1.3
                                            detection_confidence > confidence_threshold and
                                            len(box_points) >= min_lidar_points):
                                            high_risk = True
                                            has_valid_detection = True
                                            print(f"  >>> {CLASS_NAMES[cls_id]} at {min_dist:.2f}m, TTC={ttc:.2f}s, Conf={detection_confidence:.2f}")

                                        detections.append({
                                            'cls_id': cls_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                            'confidence': detection_confidence, 'distance': min_dist,
                                            'ttc': ttc, 'lidar_points': len(box_points)
                                        })
                                except:
                                    continue

        min_distance_to_obstacle = min_distance if min_distance < float('inf') else float('inf')

        aeb_history.append(high_risk and has_valid_detection)
        should_brake = sum(aeb_history) >= 1  # Changed from 2 - more aggressive, single detection triggers AEB

        if should_brake and has_valid_detection:
            if aeb_state == AEB_STATE_INACTIVE:
                aeb_state = AEB_STATE_WARNING
                warning_active = True
                print("⚠️ AEB WARNING")
            elif aeb_state == AEB_STATE_WARNING:
                aeb_state = AEB_STATE_BRAKING
                print("🛑 AEB ACTIVATED")
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

        frame_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

        # Draw LiDAR points
        for pt_2d, pt_3d in debug_lidar_points:
            x, y = int(pt_2d[0]), int(pt_2d[1])
            pygame.draw.circle(frame_surface, (0, 255, 0), (x, y), 1)

        # Draw detections
        for det in detections:
            cls_id = det['cls_id']
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            confidence = det['confidence']
            distance = det['distance']
            ttc = det['ttc']
            lidar_pts = det.get('lidar_points', 0)

            if cls_id in RELEVANT_CLASSES:
                color = (255, 255, 0) if cls_id == 0 else (255, 0, 0)  # Yellow for pedestrians
                label = CLASS_NAMES.get(cls_id, 'Object')

                pygame.draw.rect(frame_surface, color, (x1, y1, x2-x1, y2-y1), 3)
                font = pygame.font.SysFont(None, 22)
                text = font.render(f"{label} ({confidence:.2f})", True, color)
                frame_surface.blit(text, (x1, y1 - 20))
                info_text = font.render(f"{distance:.1f}m | {lidar_pts}pts | TTC:{ttc:.1f}s", True, color)
                frame_surface.blit(info_text, (x1, y1 - 45))

        # HUD
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Speed: {speed_kmh:.1f} km/h", True, (255, 255, 255))
        frame_surface.blit(speed_text, (10, H - 50))

        if pedestrian_detected and min_distance_to_obstacle < float('inf'):
            dist_text = font.render(f"Pedestrian: {min_distance_to_obstacle:.1f}m", True, (255, 255, 0))
            frame_surface.blit(dist_text, (10, H - 25))

        if aeb_state == AEB_STATE_BRAKING:
            warn = font.render("🛑 EMERGENCY BRAKING 🛑", True, (255, 0, 0))
            frame_surface.blit(warn, (W // 2 - warn.get_width() // 2, 10))
        elif aeb_state == AEB_STATE_WARNING:
            warn = font.render("⚠️ COLLISION WARNING ⚠️", True, (255, 255, 0))
            frame_surface.blit(warn, (W // 2 - warn.get_width() // 2, 10))

    except Exception as e:
        print(f"❌ Camera error: {e}")

# ------------------- Main -------------------
def main():
    global simulation_active, aeb_state, aeb_active, emergency_brake, min_distance_to_obstacle
    global pedestrian_crossing, pedestrian_triggered

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CARLA AEB – Scenario 7: Pedestrian Crossing (FIXED)")
    clock = pygame.time.Clock()

    print("="*60)
    print("CARLA AEB - Scenario 7: Car-to-Pedestrian (CPNA) - FIXED")
    print("="*60)

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    vehicle = None
    camera = None
    lidar = None
    pedestrian = None
    world = None
    original_settings = None

    try:
        print("⏳ Connecting to CARLA...")
        time.sleep(2)

        print("\n📋 Loading OpenDRIVE map (Scenario 7: Pedestrian Crossing)...")
        opendrive_content = create_pedestrian_road_map()
        world = client.generate_opendrive_world(
            opendrive_content,
            carla.OpendriveGenerationParameters(
                vertex_distance=2.0,
                max_road_length=50.0,
                wall_height=1.0,
                additional_width=0.6,
                smooth_junctions=True,
                enable_mesh_visibility=True
            )
        )
        print("✅ Map loaded")

        # Synchronous mode setup
        settings = world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)
        print("✅ Synchronous mode enabled")

        world.set_weather(carla.WeatherParameters.ClearNoon)
        print("✅ Weather set")

        # Get spawn points
        carla_map = world.get_map()
        map_spawn_points = carla_map.get_spawn_points()
        print(f"Found {len(map_spawn_points)} spawn points")

        if len(map_spawn_points) == 0:
            print("⚠️ No spawn points - using fallback")
            map_spawn_points = [carla.Transform(carla.Location(x=5.0, y=0.0, z=0.2), carla.Rotation(yaw=0.0))]

        bp_lib = world.get_blueprint_library()

        # Spawn ego vehicle
        print("\n🚗 Spawning ego vehicle...")
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

        # Spawn pedestrian on right shoulder ~40m ahead
        print("\n🚶 Spawning pedestrian (using direct velocity control)...")

        # Find all pedestrian blueprints
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        if not walker_bps:
            raise RuntimeError("No pedestrian blueprints found!")

        # Pick an adult pedestrian (not child)
        adult_walkers = [bp for bp in walker_bps if 'child' not in bp.id.lower()]
        if not adult_walkers:
            adult_walkers = list(walker_bps)

        pedestrian_bp = random.choice(adult_walkers)
        print(f"   Selected pedestrian: {pedestrian_bp.id}")

        # Get ego position and place pedestrian 150m ahead on right shoulder
        ego_location = vehicle.get_transform().location
        pedestrian_start_location = carla.Location(
            x=ego_location.x + 150.0,
            y=ego_location.y + 2.5,  # Right shoulder (right side of road)
            z=0.2
        )
        pedestrian_transform = carla.Transform(
            pedestrian_start_location,
            carla.Rotation(yaw=-90.0)  # Facing left (ready to cross)
        )

        # Try spawning at multiple heights if collision occurs
        spawn_attempts = [0.2, 0.5, 1.0, 1.5]
        for z_offset in spawn_attempts:
            try:
                pedestrian_transform.location.z = z_offset
                pedestrian = world.spawn_actor(pedestrian_bp, pedestrian_transform)
                print(f"✅ Pedestrian spawned at ({pedestrian_start_location.x:.1f}, {pedestrian_start_location.y:.1f}, {z_offset:.1f})")
                break
            except Exception as e:
                if z_offset == spawn_attempts[-1]:
                    raise RuntimeError(f"Failed to spawn pedestrian after {len(spawn_attempts)} attempts")
                continue

        if pedestrian is None:
            raise RuntimeError("Failed to spawn pedestrian")

        # No AI controller needed - we'll use WalkerControl API
        print("   Using WalkerControl API (no AI controller)")

        # Stabilize
        for _ in range(10):
            world.tick()
        time.sleep(0.2)

        ped_pos = pedestrian.get_transform().location
        print(f"   Pedestrian position: ({ped_pos.x:.1f}, {ped_pos.y:.1f}, {ped_pos.z:.1f})")

        # Calculate crossing target (left side of road)
        pedestrian_target_y = ped_pos.y - 5.5  # Cross to left side
        print(f"   Crossing target Y: {pedestrian_target_y:.1f}")

        # Verify ego and pedestrian positions
        ego_pos = vehicle.get_transform().location
        initial_distance = math.sqrt((ego_pos.x - ped_pos.x)**2 + (ego_pos.y - ped_pos.y)**2)
        print(f"   Initial distance from ego to pedestrian: {initial_distance:.1f}m")

        # Setup sensors
        print("\n📷 Setting up sensors...")
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(W))
        cam_bp.set_attribute("image_size_y", str(H))
        cam_bp.set_attribute("fov", str(FOV))
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))

        cam_tf = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.8),
            carla.Rotation(pitch=-3.0, yaw=0.0, roll=0.0)
        )
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(LIDAR_RANGE))
        lidar_bp.set_attribute("channels", str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("points_per_second", str(LIDAR_PPS))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROT_HZ))
        lidar_bp.set_attribute("sensor_tick", str(FIXED_DT))
        lidar_tf = carla.Transform(carla.Location(z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        print("✅ Sensors created")

        print("\n🤖 Loading YOLO...")
        model = YOLO("yolov8n.pt")
        print("✅ YOLO loaded")

        camera.listen(lambda img: camera_callback(img, model, vehicle, camera, lidar))
        lidar.listen(lidar_callback)

        for _ in range(20):
            world.tick()
        time.sleep(0.5)

        print("\n" + "="*60)
        print("🎮 SYSTEM READY! (FIXED VERSION)")
        print("="*60)
        print(f"✓ Pedestrian waiting at ({ped_pos.x:.1f}, {ped_pos.y:.1f})")
        print(f"✓ Initial distance: {initial_distance:.1f}m")
        print(f"✓ Auto-trigger distance: {PEDESTRIAN_TRIGGER_DISTANCE}m")
        print(f"✓ Crossing speed: {PEDESTRIAN_WALKING_SPEED * 3.6:.1f} km/h (fast walk)")
        print("✓ Using WalkerControl API (no AI navigation)")
        print("✓ AEB configured to detect pedestrians")
        print("\n🎮 Controls:")
        print("   W/Up : Drive forward (control your speed)")
        print("   P : Manually trigger pedestrian crossing (for testing)")
        print("   Q/ESC : Quit")
        print("\n📊 Debug info will show distance every 2 seconds")
        print("="*60 + "\n")

        simulation_active = True
        running = True
        frame_count = 0
        last_distance_print = 0

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif e.key == pygame.K_p and not pedestrian_triggered:
                        # Manual trigger for testing
                        print(f"\n🔧 MANUAL PEDESTRIAN TRIGGER (P key pressed)")
                        try:
                            ego_loc = vehicle.get_transform().location
                            ped_loc = pedestrian.get_transform().location
                            distance = math.sqrt((ego_loc.x - ped_loc.x)**2 + (ego_loc.y - ped_loc.y)**2)
                            print(f"   Current distance: {distance:.1f}m")

                            # Use WalkerControl (correct API for pedestrians)
                            pedestrian_triggered = True
                            pedestrian_crossing = True
                            print(f"✅ Pedestrian crossing manually triggered (WalkerControl)")
                        except Exception as e:
                            print(f"❌ Manual trigger failed: {e}")

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
                aeb_state = AEB_STATE_INACTIVE
                aeb_active = False
                emergency_brake = False
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle = 0.7
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                brake = 1.0

            # Check distance to pedestrian and trigger crossing
            frame_count += 1

            if not pedestrian_triggered and pedestrian is not None:
                try:
                    ego_loc = vehicle.get_transform().location
                    ped_loc = pedestrian.get_transform().location
                    distance_to_ped = math.sqrt(
                        (ego_loc.x - ped_loc.x)**2 +
                        (ego_loc.y - ped_loc.y)**2
                    )

                    # Print distance every 2 seconds (40 frames at 20Hz)
                    if frame_count - last_distance_print > 40:
                        print(f"Distance to pedestrian: {distance_to_ped:.1f}m (trigger at {PEDESTRIAN_TRIGGER_DISTANCE}m)")
                        last_distance_print = frame_count

                    if distance_to_ped < PEDESTRIAN_TRIGGER_DISTANCE:
                        print(f"\n{'='*60}")
                        print(f"🚶 PEDESTRIAN CROSSING TRIGGERED!")
                        print(f"   Distance: {distance_to_ped:.1f}m")
                        print(f"   Ego at: ({ego_loc.x:.1f}, {ego_loc.y:.1f})")
                        print(f"   Ped at: ({ped_loc.x:.1f}, {ped_loc.y:.1f})")
                        print(f"{'='*60}\n")

                        try:
                            # Use WalkerControl (correct API for pedestrians)
                            # Will be applied every frame in main loop below
                            pedestrian_triggered = True
                            pedestrian_crossing = True
                            print(f"✅ Pedestrian crossing triggered (WalkerControl at {PEDESTRIAN_WALKING_SPEED * 3.6:.1f} km/h)")
                        except Exception as e:
                            print(f"❌ Failed to trigger pedestrian: {e}")

                except Exception as e:
                    print(f"❌ Error in distance check: {e}")

            # Apply pedestrian control if crossing
            if pedestrian_crossing and pedestrian is not None:
                try:
                    # Create and apply WalkerControl every frame
                    walker_control = carla.WalkerControl()
                    walker_control.direction = carla.Vector3D(x=0.0, y=-1.0, z=0.0)  # Normalized: left
                    walker_control.speed = PEDESTRIAN_WALKING_SPEED  # m/s
                    pedestrian.apply_control(walker_control)
                except Exception as e:
                    print(f"❌ Walker control error: {e}")

            # AEB control
            if aeb_state == AEB_STATE_BRAKING:
                v = vehicle.get_velocity()
                speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
                ttc = min_distance_to_obstacle / speed if speed > 0.1 else float('inf')
                # Very aggressive braking - minimum 0.8, scales with TTC
                brake_force = min(1.0, max(0.8, 5.0 - 1.5 * ttc))
                ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake_force, reverse=False)
            else:
                ctrl = carla.VehicleControl(
                    throttle=max(0.0, min(1.0, throttle)),
                    steer=max(-1.0, min(1.0, steer)),
                    brake=max(0.0, min(1.0, brake)),
                    reverse=reverse
                )

            vehicle.apply_control(ctrl)
            world.tick()

            if frame_surface:
                screen.blit(frame_surface, (0, 0))
                font = pygame.font.SysFont(None, 20)
                for i, line in enumerate(CONTROLS_TEXT):
                    screen.blit(font.render(line, True, (255, 255, 255)), (10, 10 + i*20))

                # Show pedestrian status
                try:
                    ego_loc = vehicle.get_transform().location
                    ped_loc = pedestrian.get_transform().location
                    current_distance = math.sqrt((ego_loc.x - ped_loc.x)**2 + (ego_loc.y - ped_loc.y)**2)

                    if not pedestrian_triggered:
                        status = font.render(f"🚶 Pedestrian waiting... Distance: {current_distance:.1f}m (Trigger: {PEDESTRIAN_TRIGGER_DISTANCE}m)", True, (255, 255, 255))
                        trigger_hint = font.render("Press P to manually trigger crossing", True, (200, 200, 200))
                        screen.blit(trigger_hint, (10, H - 100))
                    else:
                        status = font.render(f"🚶 PEDESTRIAN CROSSING! Distance: {current_distance:.1f}m", True, (255, 255, 0))
                    screen.blit(status, (10, H - 75))
                except:
                    pass

                pygame.display.flip()
            else:
                screen.fill((0, 0, 0))
                font = pygame.font.SysFont(None, 32)
                msg = font.render("Initializing... Press Q to quit", True, (255, 255, 255))
                screen.blit(msg, (W//2 - msg.get_width()//2, H//2))
                pygame.display.flip()

            clock.tick(60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        simulation_active = False
        time.sleep(0.5)

        try:
            if camera:
                camera.stop()
            if lidar:
                lidar.stop()
            if world and original_settings:
                world.apply_settings(original_settings)
        except:
            pass

        for actor in [camera, lidar, pedestrian, vehicle]:
            if actor:
                try:
                    actor.destroy()
                except:
                    pass

        pygame.quit()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    print("\n🚀 Starting CARLA AEB - Scenario 7 (Pedestrian Crossing - FIXED)...\n")
    main()
    print("\n✅ Program terminated\n")
