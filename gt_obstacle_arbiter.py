#!/usr/bin/env python3
import json
import math
import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates

LANE_TOPIC = "/cmd/lane"
CMD_TOPIC  = "/automobile/command"
MODEL_TOPIC = "/gazebo/model_states"

class GTObstacleArbiter:
	def __init__(self):
    	# --- Params ---
    	self.lane_topic = rospy.get_param("~lane_topic", LANE_TOPIC)
    	self.cmd_topic = rospy.get_param("~cmd_topic", CMD_TOPIC)
    	self.model_topic = rospy.get_param("~model_topic", MODEL_TOPIC)
    	# Identify ego + obstacles by name keywords (edit if needed)
    	self.ego_keywords = rospy.get_param("~ego_keywords", ["automobile", "ego"])
    	self.vehicle_keywords = rospy.get_param("~vehicle_keywords", ["car", "vehicle", "sedan", "truck", "bus", "van", "ped", "person", "p>

    	# Detection gates (meters, ego frame)
    	self.stop_dist = float(rospy.get_param("~stop_dist", 2.5))    	# stop if obstacle ahead closer than this
    	self.lateral_gate = float(rospy.get_param("~lateral_gate", 0.7)) # consider "in lane" if |y| < this

    	# Command format
    	self.steer_key = rospy.get_param("~steer_key", "steerAngle")

    	# Safety timeouts
    	self.hz = float(rospy.get_param("~hz", 20))
    	self.lane_timeout = float(rospy.get_param("~lane_timeout", 0.5))

    	# --- State ---
    	self.last_lane = {"speed": 0.0, self.steer_key: 0.0}
    	self.last_lane_time = rospy.Time(0)
    	self.models = None

    	# --- ROS IO ---
    	self.pub_cmd = rospy.Publisher(self.cmd_topic, String, queue_size=10)
    	rospy.Subscriber(self.lane_topic, String, self.cb_lane, queue_size=10)
    	rospy.Subscriber(self.model_topic, ModelStates, self.cb_models, queue_size=1)

    	rospy.Timer(rospy.Duration(1.0 / self.hz), self.loop)
    	rospy.loginfo("[gt_obstacle_arbiter] ready: /cmd/lane + /gazebo/model_states -> /automobile/command")

	def cb_lane(self, msg: String):
    	try:
        	data = json.loads(msg.data)
        	if "speed" in data:
            	self.last_lane["speed"] = float(data["speed"])
        	if self.steer_key in data:
            	self.last_lane[self.steer_key] = float(data[self.steer_key])
        	elif "steer" in data:
            	self.last_lane[self.steer_key] = float(data["steer"])
        	self.last_lane_time = rospy.Time.now()
    	except Exception:

        	pass

	def cb_models(self, msg: ModelStates):
    	self.models = msg

	def pub_speed(self, v):
    	self.pub_cmd.publish(String(data=json.dumps({"action": "1", "speed": float(v)})))

	def pub_steer(self, a):
    	self.pub_cmd.publish(String(data=json.dumps({"action": "2", self.steer_key: float(a)})))

	@staticmethod
	def yaw_from_quat(q):
    	siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
    	cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    	return math.atan2(siny_cosp, cosy_cosp)

	def find_ego_index(self, names):
    	for i, n in enumerate(names):
        	ln = n.lower()
        	if any(k in ln for k in self.ego_keywords):
            	return i
    	return None

	def is_vehicle_name(self, name):
    	ln = name.lower()
    	return any(k in ln for k in self.vehicle_keywords)

	def nearest_obstacle_ahead(self):
    	if self.models is None:
        	return None

    	names = self.models.name
    	poses = self.models.pose

    	ego_i = self.find_ego_index(names)
    	if ego_i is None:
        	return None

    	ego_p = poses[ego_i].position
    	ego_q = poses[ego_i].orientation
    	ego_yaw = self.yaw_from_quat(ego_q)

    	cy = math.cos(-ego_yaw)
    	sy = math.sin(-ego_yaw)

    	best = None
    	best_x = 1e9

    	for i, n in enumerate(names):
        	if i == ego_i:
            	continue
        	if not self.is_vehicle_name(n):
            	continue

        	p = poses[i].position
        	dx = p.x - ego_p.x
        	dy = p.y - ego_p.y

        	ex = cy*dx - sy*dy   # forward
        	ey = sy*dx + cy*dy   # left

        	if ex <= 0.0:
            	continue
        	if abs(ey) > self.lateral_gate:
            	continue

        	if ex < best_x:
            	best_x = ex
            	best = {"x": ex, "y": ey, "name": n}

    	return best

	def loop(self, _evt):
    	now = rospy.Time.now()

    	# If lane command stale, stop
    	if (now - self.last_lane_time).to_sec() > self.lane_timeout:
        	self.pub_speed(0.0)
        	self.pub_steer(self.last_lane.get(self.steer_key, 0.0))

	def loop(self, _evt):
    	now = rospy.Time.now()

    	# If lane command stale, stop
    	if (now - self.last_lane_time).to_sec() > self.lane_timeout:
        	self.pub_speed(0.0)
        	self.pub_steer(self.last_lane.get(self.steer_key, 0.0))
        	return

    	lane_speed = float(self.last_lane["speed"])
    	lane_steer = float(self.last_lane[self.steer_key])

    	obs = self.nearest_obstacle_ahead()

    	# No obstacle -> pass through
    	if obs is None:
        	self.pub_speed(lane_speed)
        	self.pub_steer(lane_steer)
        	return

    	ex = obs["x"]
    	ey = obs["y"]

    	# Obstacle within stop_dist and in lane -> full stop until it moves
    	if ex <= self.stop_dist:
        	rospy.loginfo_throttle(1.0, f"[arbiter] obstacle {obs['name']} at x={ex:.2f}m y={ey:.2f}m - FULL STOP")
        	self.pub_speed(0.0)
        	self.pub_steer(lane_steer)
        	return

    	# Obstacle ahead but beyond stop_dist -> pass through
    	self.pub_speed(lane_speed)
    	self.pub_steer(lane_steer)

def main():
	rospy.init_node("gt_obstacle_arbiter")
	GTObstacleArbiter()
	rospy.spin()

if __name__ == "__main__":
	main()

