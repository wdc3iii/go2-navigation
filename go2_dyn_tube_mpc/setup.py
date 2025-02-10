import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'go2_dyn_tube_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        # this is how you expose data directories in your package
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.obk"))),
        (os.path.join("share", package_name, "rviz"), glob(os.path.join("rviz", "*.rviz"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wcompton',
    maintainer_email='wdc3iii@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dyn_tube_mpc = go2_dyn_tube_mpc.dyn_tube_mpc_node:main',
            'high_level_planner = go2_dyn_tube_mpc.high_level_planner_node:main'
            'trajectory_tracker = go2_dyn_tube_mpc.trajectory_tracker_node:main'
        ],
    },
)
