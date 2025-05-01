import csv
import pandas as pd
from racing_mppi.Track.CubicSpline2D import CubicSpline2D
from racing_mppi.path_generate import make_csv_paths, make_side_lane

class LaneInfo:
    def __init__(self, s=0.0, left_width=0.0, right_width=0.0):
        self.s = s
        self.left_width = left_width
        self.right_width = right_width

class Waypoint:
    def __init__(self, x=0.0, y=0.0, v=0.0, vm=0.0):
        self.x = x
        self.y = y
        self.v = v
        self.vm = vm

class WaypointArray:
    def __init__(self):
        self.wp = []

class Track:
    def __init__(self, centerline_filename, interval=0.1):
        self.csp = CubicSpline2D()
        self.global_path = WaypointArray()
        self.lane = []

        self.interval = interval
        self.center_path, _, _ = make_csv_paths(centerline_filename, DL=self.interval, offset=False)
        self.right_lane, self.left_lane = make_side_lane(self.center_path, lane_width=1.1)
        self.clac_width()
        self.set_cubic_spline()

    def clac_width(self):
        s = 0.0
        print(len(self.center_path), len(self.right_lane), len(self.left_lane))
        for i in range(len(self.center_path)):
            x = self.center_path[i][0]
            y = self.center_path[i][1]

            right_x = self.right_lane[i][0]
            right_y = self.right_lane[i][1]
            right_width = ((right_x - x) ** 2 + (right_y - y) ** 2) ** 0.5

            left_x = self.left_lane[i][0]
            left_y = self.left_lane[i][1]
            left_width = ((left_x - x) ** 2 + (left_y - y) ** 2) ** 0.5

            info = LaneInfo(s, left_width, right_width)
            self.lane.append(info)
            s += self.interval
        print("track length:", s)
    # def read_centerline(self, filename):
    #     """Reads the centerline data from a CSV file."""
    #     print(f"Reading centerline from: {filename}")
    #     # with open(filename, 'r') as file:
    #     #     csv_reader = csv.reader(file)
    #     #     for row in csv_reader:
    #     #         if not row:
    #     #             continue
    #     #         waypoint = Waypoint(float(row[0]), float(row[1]), 0.0, 0.0)
    #     #         self.global_path.wp.append(waypoint)

    #     df = pd.read_csv(filename)
    #     df.columns = df.columns.str.lstrip('#').str.strip()
    #     df.columns = df.columns.str.lstrip(' ').str.strip()
    #     s = 0.0
    #     for i in range(len(df)):
    #         x_m = df.loc[i, 'x_m']
    #         y_m = df.loc[i, 'y_m']
    #         w_tr_right_m = df.loc[i, 'w_tr_right_m']
    #         w_tr_left_m = df.loc[i, 'w_tr_left_m']

    #         waypoint = Waypoint(x_m, y_m, 0.0, 0.0)
    #         self.global_path.wp.append(waypoint)

    #         if i == 0:
    #             info = LaneInfo(s, w_tr_right_m, w_tr_left_m)
    #         else:
    #             s += ((x_m - df.loc[i - 1, 'x_m']) ** 2 + (y_m - df.loc[i - 1, 'y_m']) ** 2) ** 0.5
    #             info = LaneInfo(s, w_tr_right_m, w_tr_left_m)
    #         self.lane.append(info)

    #     print(f"Global path size: {len(self.global_path.wp)}")

    # def read_width_info(self, filename):
    #     """Reads lane width information from a CSV file."""
    #     print(f"Reading width info from: {filename}")
    #     with open(filename, 'r') as file:
    #         csv_reader = csv.reader(file)
    #         for row in csv_reader:
    #             if not row:
    #                 continue
    #             info = LaneInfo(float(row[0]), float(row[1]), float(row[2]))
    #             self.lane.append(info)

    #     print(f"Lane info size: {len(self.lane)}")

    def set_cubic_spline(self):
        """Sets up the cubic spline for the track."""
        sample_path = [wp for wp in self.global_path.wp]
        self.csp.setCubicSpline2D(sample_path)

    def unwrap_spline_length(self, s_max, spline_length):
        """Unwraps spline length to ensure it is within a valid range."""
        if spline_length < -s_max / 2:
            spline_length += s_max
        elif spline_length > s_max / 2:
            spline_length -= s_max
