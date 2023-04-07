# Investigating Deep Reinnforcement Learning for Quadrotor Planning and Control

## Andrew Garrett, Ramya Muthukrishnan, Philip Sieg

For our ESE650 Final Project, we plan to...

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

 - [] Figure out tensorboard vs. wandb results tracking
 - [] 