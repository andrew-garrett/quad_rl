#################### IMPORTS ####################
#################################################


import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


#################### GLOBAL VARIABLES ####################
##########################################################


DEFAULT_DRONES = DroneModel("cf2x") # gym-pybullet-drones model
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb") # Physics("pyb_gnd_drag_dw")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_H = 10.0
DEFAULT_OFFSET = 1.0


def initialize_run(
    trajectory,
    AGGR_PHY_STEPS,
    H=DEFAULT_H,
    OFFSET=DEFAULT_OFFSET,
    drone=DEFAULT_DRONES,
    physics=DEFAULT_PHYSICS,
    vision=DEFAULT_VISION,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    aggregate=DEFAULT_AGGREGATE,
    obstacles=DEFAULT_OBSTACLES,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    colab=DEFAULT_COLAB
):
    num_drones = trajectory.params["num_drones"]**3
    if num_drones == 1:
        INIT_XYZS = np.array([np.array([0, 0, H]) for k in range(num_drones)])
    else:
        cube_dim = int(np.floor(num_drones**(1/3)))
        INIT_XYZS = OFFSET*np.indices((cube_dim, cube_dim, cube_dim)).reshape((3, num_drones)).T
        INIT_XYZS[:, -1] += H
        INIT_XYZS += trajectory.update(0.)["x"]
    
    if "rpy0" in trajectory.params.keys():
        if trajectory.params["ax"] == "r":
            INIT_RPYS = np.array([[trajectory.params["rpy0"]*np.pi/180., 0., 0.] for k in range(num_drones)])
        elif trajectory.params["ax"] == "p":
            INIT_RPYS = np.array([[0., trajectory.params["rpy0"]*np.pi/180., 0.] for k in range(num_drones)])
        else:
            INIT_RPYS = np.array([[0., 0., trajectory.params["rpy0"]*np.pi/180.] for k in range(num_drones)])
    else:
        INIT_RPYS = np.array([[0., 0., 0.] for k in range(num_drones)])

    #### Create the environment with or without video capture ##
    if vision: 
        env = VisionAviary(drone_model=drone,
                           num_drones=num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=physics,
                           neighbourhood_radius=10,
                           freq=simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=gui,
                           record=record_video,
                           obstacles=obstacles
                           )
    else: 
        env = CtrlAviary(drone_model=drone,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=physics,
                         neighbourhood_radius=10,
                         freq=simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )
        
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
        num_drones=env.NUM_DRONES,
        output_folder=f"{trajectory.root}sim_data",
        colab=colab
    )
    
    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=drone) for k in range(env.NUM_DRONES)]

    return env, logger, ctrl



def render_markers(
    points, env, xyz0_traj=None, obs=None, 
    trajSphereId=None, flightSphereId=None, 
    target_pos=None
):
    """
    Render Waypoint, Trajectory, and Flight Path Markers in the Pybullet Environment
    """
    sphereRadius = 0.05
    if xyz0_traj is not None:
        waypointSphereId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, 
            rgbaColor=[1, 0, 0, 1],
            radius=sphereRadius
        )
        plt_num_wpts = min(100, points.shape[0])
        plt_wpt_inds = np.linspace(0, points.shape[0] - 1, num=plt_num_wpts).astype("int")
        for wpt_ind in plt_wpt_inds:
            for k in range(env.NUM_DRONES):
                p.createMultiBody(
                    baseMass=0,
                    baseInertialFramePosition=[0, 0, 0],
                    baseVisualShapeIndex=waypointSphereId, 
                    basePosition=env.INIT_XYZS[k, :] + points[wpt_ind] - xyz0_traj,
                    useMaximalCoordinates=1
                )
    if trajSphereId is None:
        trajSphereId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, 
            rgbaColor=[0, 0, 1, 1],
            radius=sphereRadius*0.2
        )
        flightSphereId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, 
            rgbaColor=[0, 1, 0, 1],
            radius=sphereRadius*0.2
        )
        return trajSphereId, flightSphereId
    
    if obs is not None:
        for k in range(env.NUM_DRONES):
            p.createMultiBody(
                baseMass=0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=flightSphereId, 
                basePosition=obs[str(k)]["state"][:3],
                useMaximalCoordinates=1
            )
            p.createMultiBody(
                baseMass=0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=trajSphereId, 
                basePosition=target_pos[k],
                useMaximalCoordinates=1
            )


def get_control(
    env, ctrl, target_state, 
    position_noise_model, velocity_noise_model, 
    action, obs, CTRL_EVERY_N_STEPS
):
    
    """
    Compute the controls in RPM
    """
    pos_noise = position_noise_model.normal(loc=0., scale=0.01, size=(env.NUM_DRONES, 3))
    target_pos = env.INIT_XYZS + target_state["x"]
    target_pos += pos_noise

    vel_noise = velocity_noise_model.normal(loc=0., scale=0.001, size=(env.NUM_DRONES, 3))
    target_vel = target_state["x_dot"]
    target_vel = target_vel.reshape((1, 3)) + vel_noise
    for k in range(env.NUM_DRONES):
        if np.any(env.INIT_RPYS != 0.0):
            action[str(k)], _, _ = ctrl[k].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(k)]["state"],
                                                                target_pos=target_pos[k],
                                                                target_vel=target_vel[k],
                                                                )
        else:
            action[str(k)], _, _ = ctrl[k].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(k)]["state"],
                                                                target_pos=target_pos[k],
                                                                target_vel=target_vel[k]
                                                                )
    return target_pos, target_vel, action


#################### RUNNER ####################
################################################


def run(
        trajectory,
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        colab=DEFAULT_COLAB
        ):
    
    #### Initialize the simulation #############################
    xyz0_traj = trajectory.update(0.)["x"]
    t_duration = trajectory.t_start_vec.flatten()[-1]
    position_noise_model = np.random.default_rng()
    velocity_noise_model = np.random.default_rng()
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

    env, logger, ctrl = initialize_run(
        trajectory,
        AGGR_PHY_STEPS,
        drone=drone,
        physics=physics,
        vision=vision,
        gui=gui,
        record_video=record_video,
        plot=plot,
        user_debug_gui=user_debug_gui,
        aggregate=aggregate,
        obstacles=obstacles,
        simulation_freq_hz=simulation_freq_hz,
        control_freq_hz=control_freq_hz,
        colab=colab
    )

    if env.NUM_DRONES == 1 and gui: #### Render Waypoints
        flightSphereId, trajSphereId = render_markers(trajectory.points, env, xyz0_traj=xyz0_traj)

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {str(k): np.array([0,0,0,0]) for k in range(env.NUM_DRONES)}
    t_counter = 0
    START = time.time()

    for t_counter in range(0, int(t_duration*env.SIM_FREQ), AGGR_PHY_STEPS):
    
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Check if it looks like we are gonna crash #############
        if np.any([obs[str(k)]["state"][2] <= DEFAULT_OFFSET for k in range(env.NUM_DRONES)]):
            break

        #### Compute control at the desired frequency ##############
        if t_counter%CTRL_EVERY_N_STEPS == 0:
            target_state = trajectory.update(t_counter*env.TIMESTEP)
            target_state["x"] -= xyz0_traj # normalize
            target_pos, target_vel, action = get_control(
                env, ctrl, target_state, 
                position_noise_model, velocity_noise_model, 
                action, obs, CTRL_EVERY_N_STEPS
            )
            if env.NUM_DRONES == 1 and gui: #### Plot Trajectory and Flight
                render_markers(
                    trajectory.points, env, obs=obs, 
                    trajSphereId=trajSphereId, flightSphereId=flightSphereId, 
                    target_pos=target_pos
                )

        for k in range(env.NUM_DRONES): #### Log the simulation 
            logger.log(
                drone=k,
                timestamp=t_counter/env.SIM_FREQ,
                state=obs[str(k)]["state"],
                control=np.hstack([target_pos[k], np.zeros(3), target_vel[k], np.zeros(3)])
            )
        if t_counter%env.SIM_FREQ == 0: #### Printout
            env.render()
            if vision: #### Print matrices with the images captured by each drone 
                for k in range(env.NUM_DRONES):
                    print(obs[str(k)]["rgb"].shape, np.average(obs[str(k)]["rgb"]),
                          obs[str(k)]["dep"].shape, np.average(obs[str(k)]["dep"]),
                          obs[str(k)]["seg"].shape, np.average(obs[str(k)]["seg"])
                    )

        if gui: #### Sync the simulation
            sync(t_counter, START, env.TIMESTEP)

    env.close() #### Close the environment
    logger.save() #### Save the simulation results
         


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Trajectory Tracking Script')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=DEFAULT_VISION,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))