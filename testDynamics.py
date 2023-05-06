from copy import deepcopy
from mppi.MPPINode import get_mppi_config
from mppi.dynamics_models import AnalyticalModel, SampleLearnedModel
from matplotlib import pyplot as plt

import numpy as np

class TestDynamics():
    """Class for testing performance of dynamics models"""

    def __init__(self, model, test_states, title):
        #Dynamics model object, to be tested
        self.model = model
        # self.model2 = model2
        #Data from a single drone
        self.data = test_states
        _ , self.n = np.shape(test_states)
        self.dt = 1/48
        self.lin_accels = (self.data[3:6, 1:] - self.data[3:6, :-1])/self.dt
        self.ang_accels = (self.data[9:12, 1:] - self.data[9:12, :-1])/self.dt
        self.title = title

    def runModelStep(self, printout = False): 
        """COMPARE STEP BY STEP PREDICTIONS"""
        #preallocate prediction
        self.s_pred = np.zeros((12, self.n-1))
        #Simulate model for each time step
        for i in range(1, self.n):
            s_in = deepcopy(self.data[:12, i-1]) #previous state 
            u_in = deepcopy(self.data[12:, i]) #control action taken 
            s_out = deepcopy(self.data[:12, i])
            #Call model to get prediction
            self.s_pred[:, i-1] = self.model(s_in.reshape(1, -1), u_in.reshape(1, -1)).flatten()

            #Printing for debugging 
            if printout: 
                print("\n\n\n\n\n")
                print("In")
                print(s_in)
                print(u_in)
                print("Actual")
                print(s_out)
                print("Prediction")
                print(self.s_pred[:, i-1])
                print("---------")

    def trajPrediction(self, rollout = True):
        """COMPARE ROLLOUT OF TRAJECTORIES TO LOOK FOR COMPOUNDING ERROR"""
        self.rollout = np.zeros((12, self.n))

        self.rolled_accelerations = np.zeros((6, self.n-1))
        self.rollout[:, 0] = deepcopy(self.data[:12, 0])
        #rollout trajectory
        for i in range(1, self.n):
            if rollout:
                s_in = self.rollout[:12, i-1] #previous predicted state
            else:
                s_in = self.data[:12, i-1]

            u_in = self.data[12:, i] #requested control
            self.rollout[:, i] = self.model(deepcopy(s_in).reshape(1, -1), deepcopy(u_in).reshape(1, -1))
            self.rolled_accelerations[:, i-1] = self.model.accelerationLabels(deepcopy(s_in).reshape(1, -1), deepcopy(u_in).reshape(1, -1))


    def compareTraj(self):
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[0, :], label = 'X Actual')
        axs[0].plot(self.rollout[0, :], '--', label = 'X Predicted')
        axs[0].legend()

        axs[1].plot(self.data[1, :], label = 'Y Actual')
        axs[1].plot(self.rollout[1, :], '--', label = 'Y Predicted')
        axs[1].legend()

        
        axs[2].plot(self.data[2, :], label = 'Z Actual')
        axs[2].plot(self.rollout[2, :], '--', label = 'Z Predicted')
        axs[2].legend()
        plt.suptitle(self.title)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[3, :], label = 'V_x Actual')
        axs[0].plot(self.rollout[3, :], '--', label = 'V_x Predicted')
        axs[0].legend()

        axs[1].plot(self.data[4, :], label = 'V_y Actual')
        axs[1].plot(self.rollout[4, :], '--', label = 'V_y Predicted')
        axs[1].legend()

        axs[2].plot(self.data[5, :], label = 'V_z Actual')
        axs[2].plot(self.rollout[5, :], '--', label = 'V_z Predicted')
        axs[2].legend()
        plt.suptitle(self.title)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[6, :], label = 'Roll Actual')
        axs[0].plot(self.rollout[6, :], '--', label = 'Roll Predicted')
        axs[0].legend()

        axs[1].plot(self.data[7, :], label = 'Pitch Actual')
        axs[1].plot(self.rollout[7, :], '--', label = 'Pitch Predicted')
        axs[1].legend()

        axs[2].plot(self.data[8, :], label = 'Yaw Actual')
        axs[2].plot(self.rollout[8, :], '--', label = 'Yaw Predicted')
        axs[2].legend()
        plt.suptitle(self.title)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[9, :], label = 'Roll Rate Actual')
        axs[0].plot(self.rollout[9, :], '--', label = 'Roll Rate Predicted')
        axs[0].legend()

        axs[1].plot(self.data[10, :], label = 'Pitch Rate Actual')
        axs[1].plot(self.rollout[10, :], '--', label = 'Pitch Rate Predicted')
        axs[1].legend()

        axs[2].plot(self.data[11, :], label = 'Yaw Rate Actual')
        axs[2].plot(self.rollout[11, :], '--', label = 'Yaw Rate Predicted')
        axs[2].legend()
        plt.suptitle(self.title)
        plt.show()
    
    def compare_accels(self):
        """Method For Comparing Acceleration Rollouts"""


        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.lin_accels[0, :], label = 'X Accel Actual')
        axs[0].plot(self.rolled_accelerations[0, :], '--', label = 'X Accel Predicted')
        axs[0].legend()

        axs[1].plot(self.lin_accels[1, :], label = 'Y Accel Actual')
        axs[1].plot(self.rolled_accelerations[1, :], '--', label = 'Y Accel Predicted')
        axs[1].legend()

        
        axs[2].plot(self.lin_accels[2, :], label = 'Z Accel Actual')
        axs[2].plot(self.rolled_accelerations[2, :], '--', label = 'Z Accel Predicted')
        axs[2].legend()
        plt.suptitle(self.title)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.ang_accels[0, :], label = 'Roll Accel Actual')
        axs[0].plot(self.rolled_accelerations[3, :], '--', label = 'Roll Accel Predicted')
        axs[0].legend()

        axs[1].plot(self.ang_accels[1, :], label = ' Pitch Accel Actual')
        axs[1].plot(self.rolled_accelerations[4, :], '--', label = 'Pitch Accel Predicted')
        axs[1].legend()

        
        axs[2].plot(self.ang_accels[2, :], label = 'Yaw Accel Actual')
        axs[2].plot(self.rolled_accelerations[5, :], '--', label = 'Yaw Accel Predicted')
        axs[2].legend()
        plt.suptitle(self.title)

        plt.show()

    """VARIOUS ERROR METRICS TO PLOT AND VISUALIZE ACCURACY"""
    def linear_absolute_error(self):
        #Plot the absolute error for linear cases
        x_error = np.abs(self.data[0, 1:-1] - self.s_pred[0, :-1])
        y_error = np.abs(self.data[1, 1:-1] - self.s_pred[1, :-1])
        z_error = np.abs(self.data[2, 1:-1] - self.s_pred[2, :-1])

        vx_error = np.abs(self.data[3, 1:-1] - self.s_pred[3, :-1])
        vy_error = np.abs(self.data[4, 1:-1] - self.s_pred[4, :-1])
        vz_error = np.abs(self.data[5, 1:-1] - self.s_pred[5, :-1])

        plt.figure(1)
        plt.plot(x_error, label = "X")
        plt.plot(y_error, label = "Y")
        plt.plot(z_error, label = "Z")
        plt.title("Position Error")
        plt.xlabel("Time Step")
        plt.ylabel("Absolute Difference [m]")
        plt.legend()

        plt.figure(2)
        plt.plot(vx_error, label = "v_x")
        plt.plot(vy_error, label = "v_y")
        plt.plot(vz_error, label = "v_z")
        plt.title("Velocity Error")
        plt.xlabel("Time Step")
        plt.ylabel("Absolute Difference [m/s]")
        plt.legend()
        plt.show()

    def rotational_absolute_error(self):
        ##plot absolute error for rotational cases 
        r_error = np.abs(np.unwrap(self.data[6, 1:-1], period = 2*np.pi) - np.unwrap(self.s_pred[6, :-1], period = 2*np.pi))
        p_error = np.abs(np.unwrap(self.data[7, 1:-1], period = 2*np.pi) - np.unwrap(self.s_pred[7, :-1], period = 2*np.pi))
        y_error = np.abs(np.unwrap(self.data[8, 1:-1], period = 2*np.pi) - np.unwrap(self.s_pred[8, :-1], period = 2*np.pi))

        rr_error = np.abs(self.data[9, 1:-1] - self.s_pred[9, :-1])
        pr_error = np.abs(self.data[10, 1:-1] - self.s_pred[10, :-1])
        yr_error = np.abs(self.data[11, 1:-1] - self.s_pred[11, :-1])

        plt.figure(3)
        plt.plot(r_error, label = "Roll")
        plt.plot(p_error, label = "Pitch")
        plt.plot(y_error, label = "Yaw")
        plt.title("Orientation Error")
        plt.xlabel("Time Step")
        plt.ylabel("Absolute Difference [rad]")
        plt.legend()
        
        plt.figure(4)
        plt.plot(rr_error, label = "Roll Rate")
        plt.plot(pr_error, label = "Pitch Rate")
        plt.plot(yr_error, label = "Yaw Rate")
        plt.title("Angular Velocity Error")
        plt.xlabel("Time Step")
        plt.ylabel("Absolute Difference [rad/s]")
        plt.legend()
        plt.show()


# TEST ANALYTICAL DYNAMICS MODEL
# Create config
config = get_mppi_config()

#Create dyanmics model object
testAnalytical = AnalyticalModel(config) 
#flight_file = "./bootstrap/datasets/dyn/AGGRO_000/sim_data/save-flight-04.19.2023_21.30.37.npy"
flight_file = "PYBD2.npy"
#flight_file = "test_data_dyn2.npy"


flight_file = "C:/Users/andre/skool/quad_rl/bootstrap/datasets/mppi/dyn/DEBUG_000/sim_data/save-flight-05.05.2023_05.01.58.npy"
flight_file = "C:/Users/andre/skool/quad_rl/bootstrap/datasets/mppi/pyb/DEBUG_000/sim_data/save-flight-05.05.2023_05.06.19.npy"
test_data = np.load(flight_file)
test_state = test_data['states'][0]

"""ANALYTICAL TEST FOR DYN DATA (EXPLICIT = TRUE)"""

# testAnalyticalDYN = AnalyticalModel(config, explicit=True)
# AnalyticalTester = TestDynamics(testAnalyticalDYN, test_state[:, :], "Analytical Model DYN")
# AnalyticalTester.runModelStep(printout=False)

# # AnalyticalTester.linear_absolute_error()
# # AnalyticalTester.rotational_absolute_error()

# AnalyticalTester.trajPrediction() #WITH ROLLOUT 
# # AnalyticalTester.trajPrediction(rollout = False) #WITHOUT ROLLOUT
# AnalyticalTester.compareTraj()
# AnalyticalTester.compare_accels()

"""ANALYTICAL TEST FOR PYB DATA (EXPLICIT = FALSE)"""

# testAnalyticalPYB = AnalyticalModel(config, explicit=False)
# AnalyticalTester = TestDynamics(testAnalyticalPYB, test_state[:, :], "Analytical Model PYB")
# AnalyticalTester.runModelStep(printout=False)

# # AnalyticalTester.linear_absolute_error()
# # AnalyticalTester.rotational_absolute_error()

# # AnalyticalTester.trajPrediction() #WITH ROLLOUT 
# AnalyticalTester.trajPrediction(rollout = False) #WITHOUT ROLLOUT
# AnalyticalTester.compareTraj()
# AnalyticalTester.compare_accels()

"""NEURAL TEST FOR PYB DATA (EXPLICIT = TRUE)"""

# testNeuralDYN = SampleLearnedModel(config, explicit=True)
# NeuralTester = TestDynamics(testNeuralDYN, test_state[:, :], "Neural Model DYN")
# NeuralTester.runModelStep(printout=False)

# # NeuralTester.linear_absolute_error()
# # NeuralTester.rotational_absolute_error()

# # NeuralTester.trajPrediction() #WITH ROLLOUT 
# NeuralTester.trajPrediction(rollout = False) #WITHOUT ROLLOUT
# NeuralTester.compareTraj()
# NeuralTester.compare_accels()

"""NEURAL TEST FOR PYB DATA (EXPLICIT = FALSE)"""

# testNeuralPYB = SampleLearnedModel(config, explicit=False)
# NeuralTester = TestDynamics(testNeuralPYB, test_state[:, :], "Neural Model PYB")
# NeuralTester.runModelStep(printout=False)

# # NeuralTester.linear_absolute_error()
# # NeuralTester.rotational_absolute_error()

# # NeuralTester.trajPrediction() #WITH ROLLOUT 
# NeuralTester.trajPrediction(rollout = False) #WITHOUT ROLLOUT
# NeuralTester.compareTraj()
# NeuralTester.compare_accels()

print("end")

