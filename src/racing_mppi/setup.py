from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'racing_mppi'

def recursive_data_files(src_dir, target_dir):
    data_files = []
    for root, dirs, files in os.walk(src_dir):
        file_list = [os.path.join(root, f) for f in files]
        if file_list:
            # target_path는 install/share/racing_mppi/maps/아래의 상대경로
            target_path = os.path.join(target_dir, os.path.relpath(root, src_dir))
            data_files.append((target_path, file_list))
    return data_files

maps_data_files = recursive_data_files('maps', os.path.join('share', package_name, 'maps'))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ] + maps_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Geon-Woo Kim',
    maintainer_email='apdnxn@naver.com',
    description='TODO: Package description',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mppi_node = racing_mppi.racing_node:main',
        ],
    },
)
