import torch
import numpy as np
from typing import Tuple

import time

# import gymnasium

from racing_mppi.mppi import MPPI
from racing_mppi.racing_env import RacingEnv
from racing_mppi.lane_map_2d import LaneMap
# from envs.obstacle_map_2d import ObstacleMap

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from transforms3d import euler

class racing_controller(Node):
    def __init__(self, device=torch.device("cuda"), dtype=torch.float32) -> None:
        super().__init__("racing_controller")

        self.declare_parameter("map_name", "")
        self.declare_parameter("debug", False)
        self.declare_parameter("drive_frequency", 0.0)

        self.declare_parameter("drive_topic", "")
        self.declare_parameter("odom_topic", "")

        self.declare_parameter("model", "")

        self.debug = self.get_parameter("debug").value
        self.current_path_index = 0

        map_name = self.get_parameter("map_name").value
        model = self.get_parameter("model").value

        env = RacingEnv(map_name)
        # solver
        drive_frequency = self.get_parameter("drive_frequency").value
        self.solver = MPPI(
            horizon=25,
            num_samples=100000,
            dim_state=4,
            dim_control=2,
            delta_t=1/drive_frequency,
            dynamics=env.ki_dynamics,
            cost_func=self.cost_function,
            u_min=env.u_min,
            u_max=env.u_max,
            sigmas=torch.tensor([0.5, 0.1]),
            lambda_=1.0,
            auto_lambda=False,
        )

        # config
        self.env = env

        # cost weights
        self.Qc = 2.0  # contouring error cost
        self.Ql = 3.0  # lag error cost
        self.Qv = 50.0  # velocity cost
        self.Qo = 10000.0  # obstacle cost
        self.Qin = 0.01  # input cost
        self.Qdin = 50.5  # differential input cost

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # reference indformation (tensor)
        self.reference_path: torch.Tensor = None
        # self.obstacle_map: ObstacleMap = None
        # self.lane_map: LaneMap = None
        # self.set_cost_map(env._obstacle_map, env._lane_map)
        self.lane_map = env._lane_map

        self._curr_state = torch.zeros(4, device=self._device, dtype=self._dtype)

        self.get_logger().info(f"drive_frequency: {drive_frequency}")
        self.timer = self.create_timer(1 / drive_frequency, self.timer_callback)

        drive_topic = self.get_parameter("drive_topic").value
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'mppi_trajectories', 10)

        self._odom = Odometry()
        odom_topic = self.get_parameter("odom_topic").value
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

    def update(self, state: torch.Tensor, racing_center_path: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the controller with the current state and reference path.
        Args:
            state (torch.Tensor): current state of the vehicle, shape (4,) [x, y, yaw, v]
            racing_center_path (torch.Tensor): racing center path, shape (N, 3) [x, y, yaw]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: action sequence tensor, shape (horizon, 2) [accel, steer], state sequence tensor, shape (horizon + 1, 4) [x, y, yaw, v]
        """

        # reference
        start = time.time()
        self.reference_path, self.current_path_index = self.calc_ref_trajectory(
            state, racing_center_path, self.current_path_index, self.solver._horizon, DL=0.1, lookahead_distance=3, reference_path_interval=0.85
        )
        end = time.time()

        if self.debug:
            self.get_logger().info(f"calc_ref_trajectory time: {round((end - start) * 1000, 2)}[ms]")

        # if self.reference_path is None and self.obstacle_map is None and self.lane_map is None:
        #     raise ValueError("reference path, obstacle map, and lane map must be set before calling solve method.")

        # solve
        start = time.time()
        action_seq, state_seq = self.solver.forward(state=state)
        end = time.time()
        solve_time = end - start

        if self.debug:
            self.get_logger().info(f"solve time: {round(solve_time * 1000, 2)}[ms]")

        return action_seq, state_seq

    def get_top_samples(self, num_samples = 300) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.solver.get_top_samples(num_samples=num_samples)

    # def set_cost_map(self, obstacle_map: ObstacleMap, lane_map: LaneMap) -> None:
    #     self.obstacle_map = obstacle_map
    #     self.lane_map = lane_map

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 4) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        # info
        prev_action = info["prev_action"]
        t = info["t"] # horizon number

        # path cost
        # contouring and lag error of path
        ec = torch.sin(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) \
            -torch.cos(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])
        el = -torch.cos(self.reference_path[t, 2]) * (state[:, 0] - self.reference_path[t, 0]) \
             -torch.sin(self.reference_path[t, 2]) * (state[:, 1] - self.reference_path[t, 1])

        path_cost = self.Qc * ec.pow(2) + self.Ql * el.pow(2)

        # velocity cost
        v = state[:, 3]
        # v_target = self.reference_path[t, 3]
        # velocity_cost = self.Qv * (v - v_target).pow(2)
        velocity_cost = torch.zeros_like(v)
        velocity_cost[v >= self.env.V_MAX] = self.Qv
        velocity_cost[v < 0] = self.Qv

        # compute obstacle cost from cost map
        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        # obstacle_cost = self.obstacle_map.compute_cost(pos_batch).squeeze(1)  # (batch_size,)
        obstacle_cost = self.lane_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost = self.Qo * obstacle_cost
        # obstacle_cost = 0

        # input cost
        input_cost = self.Qin * action.pow(2).sum(dim=1)
        input_cost += self.Qdin * (action - prev_action).pow(2).sum(dim=1)

        cost = path_cost + velocity_cost + obstacle_cost + input_cost

        return cost

    def calc_ref_trajectory(self, state: torch.Tensor, path: torch.Tensor,
                            cind: int, horizon: int, DL=0.1, lookahead_distance=1.0, reference_path_interval=0.5
                            ) -> Tuple[torch.Tensor, int]:
        """
        Calculate the reference trajectory for the vehicle.

        Args:
            state (torch.Tensor): current state of the vehicle, shape (4,) [x, y, yaw, v]
            path (torch.Tensor): reference path, shape (N, 3) [x, y, yaw]
            cind (int): current index of the vehicle on the path
            horizon (int): prediction horizon
            DL (float): resolution of the path
            lookahead_distance (float): distance to look ahead
            reference_path_interval (float): interval of the reference path

        Returns:
            Tuple[torch.Tensor, int]: reference trajectory tensor, shape (horizon + 1, 4) [x, y, yaw, target_v], index of the vehicle on the path
        """

        ncourse = len(path)
        xref = torch.zeros((horizon + 1, state.shape[0]), dtype=state.dtype, device=state.device)

        # # Calculate the nearest index to the vehicle
        path_cpu = path.cpu().numpy()
        state_cpu = state.cpu().numpy()
        ind = min(range(len(path)), key=lambda i: np.hypot(path_cpu[i, 0] - state_cpu[0], path_cpu[i, 1] - state_cpu[1]))
        # Ensure the index is not less than the current index
        ind = max(cind, ind)

        # Generate the rest of the reference trajectory
        travel = lookahead_distance

        for i in range(horizon + 1):
            travel += reference_path_interval
            dind = int(round(travel / DL))

            if (ind + dind) < ncourse:
                xref[i, :3] = path[ind + dind]
                xref[i, 3] = self.env.V_MAX
            else:
                xref[i, :3] = path[-1]
                # set the target velocity to zero if the vehicle reaches the end of the path
                xref[:, 3] = 0.0

        return xref, ind

    def odom_callback(self, msg: Odometry):
        self._curr_state[0] = msg.pose.pose.position.x
        self._curr_state[1] = msg.pose.pose.position.y
        self._curr_state[2] = euler.quat2euler([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z])[2]
        self._curr_state[3] = msg.twist.twist.linear.x

    def timer_callback(self):
        time1 = time.time()
        ts = self.get_clock().now().to_msg()

        self.get_logger().info(f"current state: {self._curr_state}")
        action_seq, state_seq = self.update(self._curr_state, self.env.racing_center_path)
        # self.get_logger().info(state_seq)
        control_input = AckermannDriveStamped()
        control_input.header.stamp = ts
        control_input.drive.acceleration = action_seq[0, 0].item()
        control_input.drive.steering_angle = action_seq[0, 1].item()
        self.drive_pub.publish(control_input)

        marker_array = MarkerArray()
        marker_id = 0
        top_samples, top_weights = self.get_top_samples(num_samples=100)
        sample_cpu = top_samples.cpu().numpy()
        for i, sample in enumerate(sample_cpu):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = ts
            marker.ns = "mppi_traj"
            marker.id = i + 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.a = 0.1
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.pose.orientation.w = 1.0
            # for state in sample:
            #     point = Point()
            #     point.x = state[0].item()
            #     point.y = state[1].item()
            #     point.z = 0.0
            #     marker.points.append(point)
            points_np = sample[:, :2]
            marker.points = [Point(x=float(pt[0]), y=float(pt[1]), z=0.0) for pt in points_np]
            marker_array.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = ts
        marker.ns = "mppi_traj"
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        for state in state_seq[0]:
            point = Point()
            point.x = state[0].item()
            point.y = state[1].item()
            point.z = 0.0
            marker.points.append(point)
        marker_array.markers.append(marker)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = ts
        marker.ns = "ref_traj"
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        for state in self.reference_path:
            point = Point()
            point.x = state[0].item()
            point.y = state[1].item()
            point.z = 0.0
            marker.points.append(point)
        marker_array.markers.append(marker)

        # self.get_logger().info(f"racing_center_path: {self.env.racing_center_path.shape}")
        # for point in self.env.left_lane:
        #     marker = Marker()
        #     marker.header.frame_id = "map"
        #     marker.header.stamp = ts
        #     marker.ns = "map_path"
        #     marker.type = Marker.ARROW
        #     marker.action = Marker.ADD
        #     marker.scale.x = 0.1
        #     marker.scale.y = 0.05
        #     marker.scale.z = 0.1
        #     marker.color.a = 1.0
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     marker.id = marker_id
        #     marker_id += 1
        #     marker.pose.position.x = point[0].item()
        #     marker.pose.position.y = point[1].item()
        #     marker.pose.position.z = 0.0
        #     quat = euler.euler2quat(0.0, 0.0, point[2].item())
        #     marker.pose.orientation.x = quat[1]
        #     marker.pose.orientation.y = quat[2]
        #     marker.pose.orientation.z = quat[3]
        #     marker.pose.orientation.w = quat[0]
        #     # self.get_logger().info(f'{marker.id}')
        #     marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

        # state, _ = self.env.step(action_seq[0, :])
        self.get_logger().info(f"one step time: {round((time.time() - time1) * 1000, 2)}[ms]")

        # is_collisions = self.env.collision_check(state=state_seq)


def main(args=None):
    rclpy.init(args=args)
    controller = racing_controller()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()