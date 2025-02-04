from setuptools import find_packages, setup

package_name = 'go2_dyn_tube_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'dyn_tube_mpc = go2_dyn_tube_mpc.dyn_tube_mpc_node:main'
            'high_level_planner = go2_dyn_tube_mpc.high_level_planner:main'
        ],
    },
)
