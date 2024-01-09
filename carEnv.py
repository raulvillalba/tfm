

import numpy as np
import time
import random

import rospy
from l4data_fusion_msgs.msg import Obstacles
from l4data_fusion_msgs.msg import Obstacle
from l4vehicle_msgs.msg import VehicleState, ModelPredictiveControlCommand
from l4route_msgs.msg import RouteWithExitLanes
from l4planning_msgs.msg import Waypoints, G2Path, G2Spline, Trajectory
from carla_msgs.msg import CarlaCollisionEvent, CarlaLaneInvasionEvent

from l4map_msgs.msg import Conversions



import std_msgs.msg

from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped, PoseWithCovariance, Pose, Quaternion, Point, PoseStamped

class CarEnv:
    
    def __init__(self):

        self.pub_initialpose = rospy.Publisher("initialpose", PoseWithCovarianceStamped, queue_size=5)
        self.pub_goal = rospy.Publisher("move_base_simple/goal", PoseStamped, queue_size=5)


        self.pub_inertial_waypoints = rospy.Publisher("inertial_waypoints", Waypoints, queue_size=5)
        self.pub_accel_steer = rospy.Publisher("kia_niro_control/mpc_input", ModelPredictiveControlCommand, queue_size=5)

        self.vehicle_conversions = rospy.Subscriber("ego_vehicle_conversions", Conversions, self.conversionsCallback)
        self.lane_crossing = rospy.Subscriber("carla/ego_vehicle/lane_invasion", CarlaLaneInvasionEvent, self.laneCallback)

        # Variables
        self.vehicle_state = None
        self.data_fusion_tracks = None
        self.carla_collision = None
        self.carla_lane = None
        self.out_of_track =  False
        self.goal_reached = False

        self.goal = None
        self.done = False
        self.goal_reached = False
        
        self.lane_crossing_list = []
        self.previous_number_lanes = 0

        self.dist_to_center = 0
        self.last_s = 0
        self.lane_type = -1

        self.previous_lane = 0
        self.last_track = 0
        self.changed = False

        self.stays_stopped = 0

        self.episode_start = 0

        # initialize the node 
        rospy.init_node('carEnv', anonymous=True)

        self.observation_space = np.size(self.reset())
        


    def reset(self):
        
        self.lane_crossing_list = []
        self.previous_number_lanes = 0
        self.out_of_track =  False
        self.done = False
        self.stays_stopped = 0
       
        possible_positions = [Pose(Point(-1.8942947387695312,6.571464538574219,0),Quaternion(0,0,0.6835865367890804,0.7298694723857898)),
                    Pose(Point( -20.8867244720459,66.07513427734375,0),Quaternion(0,0,0.9998886853439396,0.014920352581899606))]

        # Publish starting point
        initialpose = PoseWithCovariance()
        pose = random.choice(possible_positions) # get random pose
        pose = possible_positions[0] 
        initialpose.pose = pose
        initialpose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
        
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        pose_to_pub = PoseWithCovarianceStamped()
        pose_to_pub.header = h
        pose_to_pub.pose = initialpose
        self.pub_initialpose.publish(pose_to_pub)

        self.vehicle_state = rospy.wait_for_message("/vehicle_state", VehicleState, timeout=None)
        self.data_fusion_tracks = rospy.wait_for_message("/data_fusion_tracks", Obstacles, timeout=None)

        accel_steer = ModelPredictiveControlCommand()
        accel_steer.left_steer_angle_command = 0
        accel_steer.acceleration_command = 0
        accel_steer.brake_deceleration_command = 10
        self.pub_accel_steer.publish(accel_steer)

        time.sleep(2)
        self.episode_start = time.time()
        return self.state_extraction()

        
        

    def step(self, action):
        # Publish new trajectory

        self.vehicle_state = rospy.wait_for_message("/vehicle_state", VehicleState, timeout=None)

        initial_point = Pose2D(self.vehicle_state.x,self.vehicle_state.y,self.vehicle_state.heading * 180 / 3.1415926535897)
        initial_speed = self.vehicle_state.velocity
        initial_heading = self.vehicle_state.heading 

        # Compute (x', y') as moving 10m and rotate 45º following the heading of the point
        theta = self.vehicle_state.heading
        x = self.vehicle_state.x
        y = self.vehicle_state.y

        accel_steer = ModelPredictiveControlCommand()
        
        # Going staight
        if action == 0: 

            d = 5
            incr_x = np.cos(theta) * d
            incr_y = np.sin(theta) * d
            p0 = Pose2D(x + incr_x / 4, y + incr_y / 4, 0)
            p1 = Pose2D(x + incr_x / 2, y + incr_y / 2, 0)
            p2 = Pose2D(x + incr_x, y + incr_y, 0)

            accel = 2

            if initial_speed < 10: # Speed limit
                accel_steer.left_steer_angle_command = 0
                accel_steer.acceleration_command = 2
                accel_steer.brake_deceleration_command = 0
            else:
                accel_steer.left_steer_angle_command = 0
                accel_steer.acceleration_command = 0
                accel_steer.brake_deceleration_command = 0

        # Brake
        elif action == 1: 
            d = 5
            incr_x = np.cos(theta) * d
            incr_y = np.sin(theta) * d
            p0 = Pose2D(x + incr_x / 4, y + incr_y / 4, 0)
            p1 = Pose2D(x + incr_x / 2, y + incr_y / 2, 0)
            p2 = Pose2D(x + incr_x, y + incr_y, 0)

            accel = -2

            accel_steer.left_steer_angle_command = 0
            accel_steer.acceleration_command = 0
            accel_steer.brake_deceleration_command = 2

        # Turn Left
        elif action == 2: 
            
            # To go straight
            d = 3
            incr_x = np.cos(theta) * d
            incr_y = np.sin(theta) * d
            p0 = Pose2D(x + incr_x / 2, y + incr_y / 2, 0)

            # Turning angle 
            incr_x = np.cos(theta + np.pi /4) * d
            incr_y = np.sin(theta + np.pi /4) * d
            p2 = Pose2D(x + incr_x, y + incr_y, 0)

            # Go back from starting point
            incr_x = np.cos(theta - np.pi / 2) * d
            incr_y = np.sin(theta - np.pi / 2)* d
            p1 = Pose2D(p2.x + incr_x / 2, p2.y + incr_y / 2, 0)

            accel = 0

            accel_steer.left_steer_angle_command = np.pi / 2
            accel_steer.acceleration_command = 0
            accel_steer.brake_deceleration_command = 0

        # Turn Right
        elif action == 3: 
            
            # To go straight
            d = 3
            incr_x = np.cos(theta) * d
            incr_y = np.sin(theta) * d
            p0 = Pose2D(x + incr_x / 2, y + incr_y / 2, 0)

            # Turning angle 
            incr_x = np.cos(theta - np.pi /4) * d
            incr_y = np.sin(theta - np.pi /4) * d
            p2 = Pose2D(x + incr_x, y + incr_y, 0)

            # Go back from starting point
            incr_x = np.cos(theta + np.pi / 2) * d
            incr_y = np.sin(theta + np.pi / 2)* d
            p1 = Pose2D(p2.x + incr_x / 2, p2.y + incr_y / 2, 0)

            accel = 0

            accel_steer.left_steer_angle_command = - np.pi / 2
            accel_steer.acceleration_command = 0
            accel_steer.brake_deceleration_command = 0
        
        # Turn Left
        elif action == 4: 
            
            # To go straight
            d = 3
            incr_x = np.cos(theta) * d
            incr_y = np.sin(theta) * d
            p0 = Pose2D(x + incr_x / 2, y + incr_y / 2, 0)

            # Turning angle 
            incr_x = np.cos(theta + np.pi /4) * d
            incr_y = np.sin(theta + np.pi /4) * d
            p2 = Pose2D(x + incr_x, y + incr_y, 0)

            # Go back from starting point
            incr_x = np.cos(theta - np.pi / 2) * d
            incr_y = np.sin(theta - np.pi / 2)* d
            p1 = Pose2D(p2.x + incr_x / 2, p2.y + incr_y / 2, 0)

            accel = 0

            accel_steer.left_steer_angle_command = np.pi / 8
            accel_steer.acceleration_command = 0
            accel_steer.brake_deceleration_command = 0

        # Turn Right
        elif action == 5: 
            
            # To go straight
            d = 3
            incr_x = np.cos(theta) * d
            incr_y = np.sin(theta) * d
            p0 = Pose2D(x + incr_x / 2, y + incr_y / 2, 0)

            # Turning angle 
            incr_x = np.cos(theta - np.pi /4) * d
            incr_y = np.sin(theta - np.pi /4) * d
            p2 = Pose2D(x + incr_x, y + incr_y, 0)

            # Go back from starting point
            incr_x = np.cos(theta + np.pi / 2) * d
            incr_y = np.sin(theta + np.pi / 2)* d
            p1 = Pose2D(p2.x + incr_x / 2, p2.y + incr_y / 2, 0)

            accel = 0

            accel_steer.left_steer_angle_command = - np.pi / 8
            accel_steer.acceleration_command = 0
            accel_steer.brake_deceleration_command = 0


        action = [p0,p1,p2,accel]

        inertial_waypoints = self.create_waypoints(action, initial_point, initial_speed, initial_heading)

        self.pub_inertial_waypoints.publish(inertial_waypoints)

        second_approach = True
        if second_approach:
            self.pub_accel_steer.publish(accel_steer)

        # Listen state: 
        self.vehicle_state = rospy.wait_for_message("/vehicle_state", VehicleState, timeout=None)
        self.data_fusion_tracks = rospy.wait_for_message("/data_fusion_tracks", Obstacles, timeout=None)

        #Check if done: either collision 
        if self.out_of_track:
            print("OUT OF TRACK!!!!!!!!!!!!!!!!")
            self.done = True
       
        reward = self.reward(action)
        return self.state_extraction(), reward, self.done, None

    def reward(self, action):
        reward = 0

        v = self.vehicle_state.velocity
        
        # Penalize 
        if self.out_of_track:
            reward -= 100
        else:
            if self.vehicle_state.lane_id == self.lane_type: 
                # Correct line
                print("Correct lane")
                if self.dist_to_center < 0.1:
                    reward += (v/0.1)/5
                else:
                    reward += (v/np.abs(self.dist_to_center))/5
                if np.abs(self.dist_to_center) > 1:
                    print("Too close to the edge")
                    reward = -2
            else:
                # incorrect line
                print("Incorrect lane")
                reward -= 4

        if v < 1:
            self.stays_stopped += 1
            reward -= 3
            if self.stays_stopped > 200:
                print("Stopped")
                self.done = True
                reward -= 50
        else:
            self.stays_stopped = 0

        return reward

    def create_waypoints(self, action, initial_point, initial_speed, initial_heading):

        inertial_waypoints = Waypoints()
        increment_of_speed = action[3]
        
        # Add inital point to list # Importante hacerlo
        inertial_waypoints.waypointList.append(initial_point)
        inertial_waypoints.velocityList.append(initial_speed)
        inertial_waypoints.curvatureList.append(initial_heading)

        previuos_point = initial_point
        speed = initial_speed
        number_of_points = 50
        for i in range(1, number_of_points + 1):
            point = self.cubic_curve(initial_point,action[0],action[1],action[2], i/number_of_points, previuos_point)
            previuos_point = point
            speed += increment_of_speed / number_of_points
            inertial_waypoints.waypointList.append(point)
            inertial_waypoints.velocityList.append(speed)
            
            inertial_waypoints.curvatureList.append(self.curvature(initial_point,action[0],action[1],action[2], i/number_of_points))

        return inertial_waypoints

    def cubic_curve(self, p0, p1, p2, p3, t, previuos_point):
         
        point_x = (1-t)*(1-t)*(1-t)*p0.x + 3*t*(1-t)*(1-t)*p1.x + 3*t*t*(1-t)*p2.x + t*t*t*p3.x
        point_y = (1-t)*(1-t)*(1-t)*p0.y + 3*t*(1-t)*(1-t)*p1.y + 3*t*t*(1-t)*p2.y + t*t*t*p3.y

        if previuos_point.x == point_x and previuos_point.y == point_y:
            point_z = p0.theta * 180 / 3.1415926535897 
        else:
            point_z = np.arctan2(point_x - previuos_point.x, point_y - previuos_point.y) * 180 / 3.1415926535897 

        return Pose2D(point_x, point_y, point_z)

    def curvature(self, p0, p1, p2, p3, t):

        dx = -3*(1-t)*(1-t)*p0.x + 3*(3*t*t-4*t+1) * p1.x + 3*(2*t-3*t*t)*p2.x + 3*t*t*p3.x
        dy = -3*(1-t)*(1-t)*p0.y + 3*(3*t*t-4*t+1) * p1.y + 3*(2*t-3*t*t)*p2.y + 3*t*t*p3.y

        ddx = 6*(1-t)*p0.x + 3*(6*t-4)*p1.x + 3*(2-6*t) * p2.x + 6*t*p3.x
        ddy = 6*(1-t)*p0.y + 3*(6*t-4)*p1.y + 3*(2-6*t) * p2.y + 6*t*p3.y

        return (dx * ddy - ddx * dy) / pow(dx * dx + dy * dy, 3/2)

        

    def rotate (self, x, y, theta, angle, distance):
        #angle = 90 + angle
        D = np.sqrt(x*x + y*y + distance*distance - 2*distance*np.sqrt(x*x+y*y)*np.cos(angle * 3.1415926535897 /180))
        delta = np.arccos((D*D + x*x + y*y - distance*distance)/(2*D*np.sqrt(x*x+y*y)))
        x_d = D * np.cos(delta + theta)
        y_d = D * np.sin(delta + theta)

        return x_d, y_d

    def conversionsCallback(self,msg):
        if len(msg.lane_coordinates) == 0:
            self.out_of_track =  True
        else:
            track_found = False
            if len(msg.lane_coordinates) > 1:
                for track in msg.lane_coordinates:
                    if track.track_id == self.last_track:
                        current_track = track   
                        track_found = True   
            
            if not track_found:
                current_track = msg.lane_coordinates[0]      

            if self.last_s <= current_track.s:
                self.lane_type = -1
            else:
                self.lane_type = 1

            self.last_s = current_track.s
            self.dist_to_center = current_track.offset
            self.out_of_track =  False

            if current_track.lane_id != self.previous_lane or current_track.track_id != self.last_track:
                self.changed = True
            else:
                self.changed = False
            self.previous_lane = current_track.lane_id 
            self.last_track = current_track.track_id

    def laneCallback(self,msg):
        self.lane_crossing_list.append(msg)
    
    def state_extraction(self):
        state = []
        state.append(self.vehicle_state.x)
        state.append(self.vehicle_state.y)
        state.append(self.vehicle_state.z)
        state.append(self.vehicle_state.heading)
        state.append(self.vehicle_state.k)
        state.append(self.vehicle_state.track_id)
        state.append(self.vehicle_state.s)
        state.append(self.vehicle_state.lane_id)
        state.append(self.vehicle_state.velocity)
        state.append(self.vehicle_state.longitudinal_velocity)
        state.append(self.vehicle_state.lateral_velocity)
        state.append(self.vehicle_state.angular_velocity)
        state.append(self.vehicle_state.acceleration)
        state.append(self.vehicle_state.longitudinal_acceleration)
        state.append(self.vehicle_state.lateral_acceleration)
        return state

    def get_state(self):
        return self.state_extraction()
