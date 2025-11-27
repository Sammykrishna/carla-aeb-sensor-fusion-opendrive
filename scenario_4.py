# main.py - CARLA AEB System with FIXED Physics-Based Braking Logic (Scenario 4: CCCscp)
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
debug_lidar_points = []  # For visualization
simulation_ready = False  # New: set when camera is ready

# ------------------- Helpers -------------------
def get_carla_version(client):
    try:
        client.get_server_version()
        return "0.10"
    except:
        return "0.9"

def safe_find(bp_lib, bp_id, carla_version):
    """Simplified safe_find per fix guide"""
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

def snap_to_driving(world, hint: carla.Location) -> carla.Transform:
    """Helper to snap to nearest drivable waypoint"""
    m = world.get_map()
    wp = m.get_waypoint(hint, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        raise RuntimeError("No drivable lane near hint location")
    tf = wp.transform
    tf.location.z += 0.3  # avoid z-fighting
    return tf

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
    points_cam = points_cam[in_front]
    return points_cam

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

def create_intersection_map():
    """FIXED OpenDRIVE map for Scenario 4 with proper junction connectors"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="CCCscp_Intersection_Fixed" version="1.4"/>
  <!-- Approaches: NOT part of the junction -->
  <road name="WestApproach" length="100.0" id="1" junction="-1">
    <link><successor elementType="junction" elementId="100" contactPoint="start"/></link>
    <planView><geometry s="0.0" x="-30.0" y="0.0" hdg="0.0" length="100.0"><line/></geometry></planView>
    <lanes><laneSection s="0.0">
      <left>
        <lane id="1" type="none" level="false"><link/>
          <width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/>
        </lane>
      </left>
      <center><lane id="0" type="none" level="false"><link/><width sOffset="0.0" a="0.0"/></lane></center>
      <right>
        <lane id="-1" type="driving" level="false"><link/>
          <width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/>
        </lane>
      </right>
    </laneSection></lanes>
  </road>
  <road name="EastExit" length="100.0" id="2" junction="-1">
    <link><predecessor elementType="junction" elementId="100" contactPoint="end"/></link>
    <planView><geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="100.0"><line/></geometry></planView>
    <lanes><laneSection s="0.0">
      <left><lane id="1" type="none" level="false"><link/><width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/></lane></left>
      <center><lane id="0" type="none" level="false"><link/><width sOffset="0.0" a="0.0"/></lane></center>
      <right><lane id="-1" type="driving" level="false"><link/><width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/></lane></right>
    </laneSection></lanes>
  </road>
  <road name="SouthApproach" length="100.0" id="3" junction="-1">
    <link><successor elementType="junction" elementId="100" contactPoint="start"/></link>
    <planView><geometry s="0.0" x="0.0" y="-30.0" hdg="1.5708" length="100.0"><line/></geometry></planView>
    <lanes><laneSection s="0.0">
      <left><lane id="1" type="none" level="false"><link/><width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/></lane></left>
      <center><lane id="0" type="none" level="false"><link/><width sOffset="0.0" a="0.0"/></lane></center>
      <right><lane id="-1" type="driving" level="false"><link/><width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/></lane></right>
    </laneSection></lanes>
  </road>
  <road name="NorthExit" length="100.0" id="4" junction="-1">
    <link><predecessor elementType="junction" elementId="100" contactPoint="end"/></link>
    <planView><geometry s="0.0" x="0.0" y="0.0" hdg="1.5708" length="100.0"><line/></geometry></planView>
    <lanes><laneSection s="0.0">
      <left><lane id="1" type="none" level="false"><link/><width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/></lane></left>
      <center><lane id="0" type="none" level="false"><link/><width sOffset="0.0" a="0.0"/></lane></center>
      <right><lane id="-1" type="driving" level="false"><link/><width sOffset="0.0" a="3.5"/><roadMark type="solid" color="white" width="0.1"/></lane></right>
    </laneSection></lanes>
  </road>
  <!-- Short in-junction connectors: ARE part of the junction -->
  <road name="ConnHoriz" length="10.0" id="101" junction="100">
    <link>
      <predecessor elementType="road" elementId="1" contactPoint="end"/>
      <successor  elementType="road" elementId="2" contactPoint="start"/>
    </link>
    <planView><geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="2.0"><line/></geometry></planView>
    <lanes><laneSection s="0.0">
      <left><lane id="1" type="none" level="false"><link/><width sOffset="0.0" a="3.5"/></lane></left>
      <center><lane id="0" type="none" level="false"><link/><width sOffset="0.0" a="0.0"/></lane></center>
      <right><lane id="-1" type="driving" level="false"><link/><width sOffset="0.0" a="3.5"/></lane></right>
    </laneSection></lanes>
  </road>
  <road name="ConnVert" length="10.0" id="102" junction="100">
    <link>
      <predecessor elementType="road" elementId="3" contactPoint="end"/>
      <successor  elementType="road" elementId="4" contactPoint="start"/>
    </link>
    <planView><geometry s="0.0" x="0.0" y="-1.0" hdg="1.5708" length="10.0"><line/></geometry></planView>
    <lanes><laneSection s="0.0">
      <left><lane id="1" type="none" level="false"><link/><width sOffset="0.0" a="3.5"/></lane></left>
      <center><lane id="0" type="none" level="false"><link/><width sOffset="0.0" a="0.0"/></lane></center>
      <right><lane id="-1" type="driving" level="false"><link/><width sOffset="0.0" a="3.5"/></lane></right>
    </laneSection></lanes>
  </road>
  <junction id="100" name="SimpleIntersection">
    <!-- West -> East via ConnHoriz -->
    <connection id="0" incomingRoad="1" connectingRoad="101" contactPoint="start">
      <laneLink from="-1" to="-1"/>
    </connection>
    <connection id="1" incomingRoad="101" connectingRoad="2"  contactPoint="end">
      <laneLink from="-1" to="-1"/>
    </connection>
    <!-- South -> North via ConnVert -->
    <connection id="2" incomingRoad="3" connectingRoad="102" contactPoint="start">
      <laneLink from="-1" to="-1"/>
    </connection>
    <connection id="3" incomingRoad="102" connectingRoad="4"  contactPoint="end">
      <laneLink from="-1" to="-1"/>
    </connection>
  </junction>
</OpenDRIVE>"""

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
        print(f"⚠️ LiDAR callback error: {e}")

def camera_callback(image, model, vehicle, camera, lidar):
    global emergency_brake, frame_surface, latest_lidar_points, aeb_active, aeb_state, warning_active
    global min_distance_to_obstacle, aeb_history, camera_ready, camera_error, detections, debug_lidar_points, simulation_ready

    if not simulation_active:
        return

    try:
        camera_ready = True
        simulation_ready = True
        detections = []
        debug_lidar_points = []

        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, :3]
        results = model.predict(img, conf=0.25, verbose=False)

        v = vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_kmh = speed * 3.6

        min_distance = float('inf')
        ttc = float('inf')
        high_risk = False
        stopping_dist = float('inf')
        has_valid_data = False
        lidar_only_min_dist = float('inf')

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
                        has_valid_data = True
                        forward_distances = filtered_cam[:, 0]
                        lidar_only_min_dist = np.min(forward_distances)
                        print(f"[LiDAR] Detected {len(filtered_cam)} points in valid zone, closest: {lidar_only_min_dist:.2f}m")

                        for r in results:
                            for box in getattr(r, "boxes", []):
                                try:
                                    cls_id = int(box.cls[0].item())
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                    if cls_id in RELEVANT_CLASSES:
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

                                            lidar_point_threshold = max(3, int(10 - min_dist * 0.5))
                                            lidar_confidence = min(1.0, len(box_points) / lidar_point_threshold)
                                            camera_confidence = 0.5
                                            if hasattr(box, 'conf') and len(box.conf) > 0:
                                                camera_confidence = float(box.conf[0].item())

                                            camera_weight = min(0.7, max(0.0, min_dist / 30.0))
                                            lidar_weight = 1.0 - camera_weight
                                            detection_confidence = (
                                                lidar_confidence * lidar_weight +
                                                camera_confidence * camera_weight
                                            )
                                            if min_dist < 15.0:
                                                detection_confidence = min(1.0, detection_confidence * 1.3)

                                            confidence_threshold = 0.15
                                            min_lidar_points = 2
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

                        if not high_risk and lidar_only_min_dist < 20.0:
                            stopping_dist = calculate_stopping_distance(speed, MAX_DECEL, REACTION_TIME)
                            ttc_lidar = calculate_ttc(lidar_only_min_dist, speed)
                            if lidar_only_min_dist < stopping_dist * 0.8 or ttc_lidar < 1.5:
                                high_risk = True
                                min_distance = lidar_only_min_dist
                                print(f"  >>> LIDAR-ONLY FALLBACK TRIGGER: Dist={lidar_only_min_dist:.2f}m, TTC={ttc_lidar:.2f}s")
                                detections.append({
                                    'cls_id': -1,
                                    'x1': W//2 - 50, 'y1': H//2 - 50,
                                    'x2': W//2 + 50, 'y2': H//2 + 50,
                                    'confidence': 0.9,
                                    'distance': lidar_only_min_dist,
                                    'ttc': ttc_lidar,
                                    'lidar_points': len(filtered_cam)
                                })

        if min_distance < float('inf'):
            min_distance_to_obstacle = min_distance
        else:
            min_distance_to_obstacle = lidar_only_min_dist

        aeb_history.append(high_risk)
        should_brake = sum(aeb_history) >= 2

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

        frame_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

        for pt_2d, pt_3d in debug_lidar_points:
            x, y = int(pt_2d[0]), int(pt_2d[1])
            distance = pt_3d[0]
            color_val = int(255 * min(1.0, distance / 30.0))
            pygame.draw.circle(frame_surface, (255 - color_val, color_val, 0), (x, y), 2)

        for det in detections:
            cls_id = det['cls_id']
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            confidence = det['confidence']
            distance = det['distance']
            ttc = det['ttc']
            lidar_pts = det.get('lidar_points', 0)

            if cls_id == -1:
                color = (255, 255, 0)
                label = "LiDAR-ONLY"
            elif cls_id in RELEVANT_CLASSES:
                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                label = CLASS_NAMES.get(cls_id, 'Object')
            else:
                continue

            pygame.draw.rect(frame_surface, color, (x1, y1, x2 - x1, y2 - y1), 3)
            font = pygame.font.SysFont(None, 24)
            text = font.render(f"{label} ({confidence:.2f})", True, color)
            frame_surface.blit(text, (x1, y1 - 20))
            info_text = font.render(f"{distance:.1f}m | {lidar_pts}pts | TTC:{ttc:.1f}s", True, color)
            frame_surface.blit(info_text, (x1, y1 - 45))

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
    global simulation_active, simulation_ready

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CARLA AEB – Scenario 4: CCCscp")
    clock = pygame.time.Clock()

    print("="*50)
    print("CARLA AEB System - Scenario 4: CCCscp")
    print("="*50)

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    time.sleep(5)

    vehicle = None
    camera = None
    lidar = None
    obstacle = None
    world = None
    original_settings = None

    try:
        print("⏳ Waiting for CARLA server...")
        try:
            server_version = client.get_server_version()
            print(f"CARLA Server Version: {server_version}")
        except:
            print("CARLA Server Version: Unknown")

        print("\nLoading FIXED OpenDRIVE map (Scenario 4: Intersection)...")
        opendrive_content = create_intersection_map()
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
        print("✅ OpenDRIVE map loaded")

        settings = world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)
        world.tick()
        time.sleep(0.1)
        print("✅ Synchronous mode enabled")

        world.set_weather(carla.WeatherParameters.ClearNoon)
        print("✅ Weather set")

        bp_lib = world.get_blueprint_library()
        print("\nSpawning ego vehicle on SouthApproach...")
        vehicle_bp = pick_vehicle_bp(bp_lib, client)

        # Use waypoint snapping for safe spawns
        ego_transform = snap_to_driving(world, carla.Location(x=0.0, y=-25.0, z=0.0))
        target_transform = snap_to_driving(world, carla.Location(x=-25.0, y=0.0, z=0.0))
        print(f"✅ Ego spawn: {ego_transform.location}")
        print(f"✅ Target spawn: {target_transform.location}")

        # Spawn ego
        vehicle = world.spawn_actor(vehicle_bp, ego_transform)
        print("✅ Ego spawned (heading north)")

        # Spawn target (obstacle)
        print("\nSpawning target vehicle on WestApproach...")
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
                if bp.has_attribute('number_of_wheels') and bp.get_attribute('number_of_wheels').as_int() == 4
            ]
            if valid_vehicles:
                obstacle_bp = random.choice(valid_vehicles)
                print(f"✅ Fallback obstacle: {obstacle_bp.id}")
            else:
                raise RuntimeError("No obstacle blueprint available")

        obstacle = world.spawn_actor(obstacle_bp, target_transform)
        print("✅ Target spawned (heading east)")

        # Setup sensors
        print("\nSetting up sensors...")
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(W))
        cam_bp.set_attribute("image_size_y", str(H))
        cam_bp.set_attribute("fov", str(FOV))
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))
        cam_bp.set_attribute("gamma", "2.2")

        cam_tf = carla.Transform(
            carla.Location(x=1.50, y=0.0, z=1.8),
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

        print("\nLoading YOLO model...")
        try:
            model = YOLO("yolov8n.pt")
            print("✅ YOLOv8n loaded")
        except Exception as e:
            print(f"❌ YOLO error: {e}")
            sys.exit(1)

        camera.listen(lambda img: camera_callback(img, model, vehicle, camera, lidar))
        print("✅ Camera listening")
        lidar.listen(lidar_callback)
        print("✅ LiDAR listening")

        # Use Traffic Manager for the crossing vehicle
        print("✅ Target vehicle will be controlled manually (no Traffic Manager)")

        print("\n=== SYSTEM READY ===")
        print("Drive NORTH with W. Target will cross from WEST.")
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
    print("Starting CARLA AEB - Scenario 4: CCCscp...")
    main()
    print("Program terminated.")
