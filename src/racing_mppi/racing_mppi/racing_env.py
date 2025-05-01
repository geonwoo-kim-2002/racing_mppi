import torch
import numpy as np
from ament_index_python.packages import get_package_share_directory

from racing_mppi.lane_map_2d import LaneMap
from racing_mppi.path_generate import make_side_lane, make_csv_paths
from racing_mppi.Track.Track import Track

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class RacingEnv:
    def __init__(self, map_name, device=torch.device("cuda"), dtype=torch.float32) -> None:
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # u: [accel, steer] (m/s2, rad)
        self.u_min = torch.tensor([-2.0, -0.25], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([2.0, 0.25], device=self._device, dtype=self._dtype)

        # model parameters
        self.L = torch.tensor(1, device=self._device, dtype=self._dtype)
        self.V_MAX = torch.tensor(3.0, device=self._device, dtype=self._dtype)

        # generate reference path
        self.dl = 0.1
        self.line_width = 1.1

        pkg_dir = get_package_share_directory("racing_mppi")
        racing_center_path, right_lane, left_lane = make_csv_paths(f"{pkg_dir}/maps/{map_name}/{map_name}_centerline.csv", DL=self.dl, offset=False)
        # self._track = Track(f"{pkg_dir}/maps/{map_name}/{map_name}_centerline.csv", interval=self.dl)

        # self.right_lane, self.left_lane = make_side_lane(racing_center_path, lane_width=self.line_width)
        # numpy array to tensor
        self.racing_center_path = torch.tensor(racing_center_path, device=self._device, dtype=self._dtype)
        self.right_lane = torch.tensor(right_lane, device=self._device, dtype=self._dtype)
        self.left_lane = torch.tensor(left_lane, device=self._device, dtype=self._dtype)

        # generate cost maps (1: lane, 2: obstacle)
        self.map_size = (161, 161)
        self.cell_size = 0.08089

        # 1: generate lane map
        self._lane_map = LaneMap(
            lane=racing_center_path,
            lane_width=self.line_width*0.8,
            map_size=self.map_size,
            cell_size=self.cell_size,
            device=self._device,
            dtype=self._dtype,
        )

    def ki_dynamics(self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 4) [x, y, theta, v]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 4) [x, y, theta, v]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = state[:, 3].view(-1, 1)
        accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dv = accel
        dtheta = v * torch.tan(steer) / self.L

        new_x = x + dx * delta_t
        new_y = y + dy * delta_t
        new_theta = angle_normalize(theta + dtheta * delta_t)
        new_v = v + dv * delta_t

        clamped_v = torch.clamp(new_v, -self.V_MAX, self.V_MAX)

        result = torch.cat([new_x, new_y, new_theta, clamped_v], dim=1)

        return result

    def st_dynamics(self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 6) [x, y, theta, v, theta_dot, beta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 6) [x, y, theta, v, theta_dot, beta]
        """

        # # Perform calculations as before
        # x = state[:, 0].view(-1, 1)
        # y = state[:, 1].view(-1, 1)
        # theta = state[:, 2].view(-1, 1)
        # v = state[:, 3].view(-1, 1)
        # theta_dot = state[:, 4].view(-1, 1)
        # beta = state[:, 5].view(-1, 1)

        # accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        # steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        # theta = angle_normalize(theta)

        # dx = v * torch.cos(theta + beta)
        # dy = v * torch.sin(theta + beta)
        # dv = accel
        # dtheta = theta_dot
        # # dtheta_dot = 

        # new_x = x + dx * delta_t
        # new_y = y + dy * delta_t
        # new_theta = angle_normalize(theta + dtheta * delta_t)
        # new_v = v + dv * delta_t

        # # Clamp x and y to the map boundary
        # # x_lim = torch.tensor(
        # #     self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        # # )
        # # y_lim = torch.tensor(
        # #     self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        # # )
        # # clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        # # clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        # clamped_v = torch.clamp(new_v, -self.V_MAX, self.V_MAX)


    #     result = torch.cat([
