from setuptools import setup, find_packages

setup(
    name='quad_rl',
    version='0.0.1',
    author='Andrew Garrett',
    description='RL and Imitation built on Pybullet and Gym',
    url='https://github.com/andrew-garrett/quad_rl',
    keywords='robotics, quadrotor, planning, control',
    python_requires='>=3.8, <3.10',
    packages=find_packages(exclude=["./gym-pybullet-drones"]),
    # install_requires=[
    #     'PyYAML',
    #     'pandas==0.23.3',
    #     'numpy>=1.14.5',
    #     'matplotlib>=2.2.0,,
    #     'jupyter'
    # ],
    # package_data={
    #     'sample': ['sample_data.csv'],
    # },
    # entry_points={
    #     'runners': [
    #         'sample=sample:main',
    #     ]
    # }
)