from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml

# import sys
# print(sys.path)

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('racing_mppi'),
        'config',
        'params.yaml'
        )
    # config_dict = yaml.safe_load(open(config, 'r'))

    mppi_node = Node(
        package='racing_mppi',
        executable='mppi_node',
        name='mppi_node',
        parameters=[config],
        output='screen',
        arguments=['--ros-args'],  # 기존 인자 뒤에
        prefix='python3 -u'        # python3 -u로 실행
    )

    ld.add_action(mppi_node)
    return ld