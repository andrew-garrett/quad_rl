# Investigating Deep Reinnforcement Learning for Quadrotor Planning and Control

## Andrew Garrett, Ramya Muthukrishnan, Philip Sieg

For our ESE650 Final Project, we implemented a Model Predictive Path Intergral (MPPI) control scheme for Crazyflie drones in the gym-pybullet-drones simulation environment. (ADD MOTIVATION FOR MPPI) 

Our project can be broken up into the following sections:

- Data Generation: Using the pybullet-drones simualtor, we generated a large and expansive amount of trajectories. These trajectories were chosen to maximize state space coverage so that a model learning the system dynamics would be robust. 
- Dynamics Models: In order to evaluate our controller, we need an accurate model $x_{i+1}f(x_i,\ u_i)$ which predicts the next state given the current state ($x$) and a desired control ($u$, RPMs of four rotors). For our drone, the kinematics are rather trivial given the velocities, so we need to only learn the dynamics (acceleration). To calculate the accelerations, we have two different approaches. The first is an standard, analytical model for a quadcopter that finds the accelerations (linear and angular) via conservation of linear and angular momentum. The second approach is via a learned model, where we train a MLP on the data collected from the simulator. This second option allows us to more easily learn complex phenomena like drag or ground effect without the need to greatly rewrite the model. Instead, we can just retrain on new simulator data 
 
- MPPI Control

# Setup instructions

This will probably take some time, so take a break while things are installing.s :)

```
conda create -n quad_rl python=3.8
conda activate quad_rl
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip install -e .
cd ..
```

TBD, but to record data videos, we may need to install ffmpeg.

If that becomes an issue, I'll create a docker image for this project.

# Example Usage

Below should generate a vizualization window of 3 quads flying along a helical path.  Then, a matplotlib window should pop up and show state and control data from all of the drones.

```
python gym-pybullet-drones/gym_pybullet_drones/examples/fly.py
```

To train an agent to hover:

```
python gym-pybullet-drones/experiments/learning/singleagent.py --cpu 2
```

To view training results, in either a separate terminal or after running the above:
```
tensorboard --logdir ./results
```

# TODO:

- Tune cost function to imrpove MPPI results 
- Implement PID as a baseline for MPPI controller 
- Create learned models for PyBullet physics with more complicated phenomena (drag, downwash, etc) 

