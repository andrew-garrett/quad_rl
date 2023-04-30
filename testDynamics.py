from mppi.MPPI_Node import get_mppi_config
from mppi.dynamics_models import AnalyticalModel, SampleLearnedModel
from matplotlib import pyplot as plt

import numpy as np

class TestDynamics():
    """Class for testing performance of dynamics models"""

    def __init__(self, model, model2, test_states, title):
        #Dynamics model object, to be tested
        self.model = model
        self.model2 = model2
        #Data from a single drone
        self.data = test_states
        _ , self.n = np.shape(test_states)

        self.title = title

    def runModelStep(self, printout = False): 
        """COMPARE STEP BY STEP PREDICTIONS"""
        #preallocate prediction
        self.s_pred = np.zeros((12, self.n-1))
        #Simulate model for each time step
        for i in range(1, self.n):
            s_in = self.data[:12, i-1] #previous state 
            u_in = self.data[12:, i] #control action taken 
            s_out = self.data[:12, i]
            #Call model to get prediction
            self.s_pred[:, i-1] = self.model2(s_in.reshape(1, -1), u_in.reshape(1, -1)).flatten()

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

    def runModelRollout(self):
        """COMPARE ROLLOUT OF TRAJECTORIES TO LOOK FOR COMPOUDNING ERROR"""
        self.rollout = np.zeros((12, self.n))

        self.rollout[:, 0] = self.data[:12, 0]

        #rollout trajectory
        for i in range(1, self.n):
            s_in = self.rollout[:, i-1] #previous predicted state
            u_in = self.data[12:, i] #requested control
            accel = self.model.accelerationLabels(s_in.reshape(1, -1), u_in.reshape(1, -1)).flatten()
            nn_accel = self.model2.accelerationLabels(s_in.reshape(1, -1), u_in.reshape(1, -1)).flatten()
            breakpoint()
            self.rollout[:, i] = self.model2(s_in.reshape(1, -1), u_in.reshape(1, -1)).flatten()

    def compareTraj(self):
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[0, :], label = 'X Actual')
        axs[0].plot(self.rollout[0, :], label = 'X Predicted')
        axs[0].legend()

        axs[1].plot(self.data[1, :], label = 'Y Actual')
        axs[1].plot(self.rollout[1, :], label = 'Y Predicted')
        axs[1].legend()

        
        axs[2].plot(self.data[2, :], label = 'Z Actual')
        axs[2].plot(self.rollout[2, :], label = 'Z Predicted')
        axs[2].legend()
        plt.suptitle(self.title)
        plt.savefig('xyz.png')

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[3, :], label = 'V_x Actual')
        axs[0].plot(self.rollout[3, :], label = 'V_x Predicted')
        axs[0].legend()

        axs[1].plot(self.data[4, :], label = 'V_y Actual')
        axs[1].plot(self.rollout[4, :], label = 'V_y Predicted')
        axs[1].legend()

        axs[2].plot(self.data[5, :], label = 'V_z Actual')
        axs[2].plot(self.rollout[5, :], label = 'V_z Predicted')
        axs[2].legend()
        plt.suptitle(self.title)
        plt.savefig('velocity.png')

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[6, :], label = 'Roll Actual')
        axs[0].plot(self.rollout[6, :], label = 'Roll Predicted')
        axs[0].legend()

        axs[1].plot(self.data[7, :], label = 'Pitch Actual')
        axs[1].plot(self.rollout[7, :], label = 'Pitch Predicted')
        axs[1].legend()

        axs[2].plot(self.data[8, :], label = 'Yaw Actual')
        axs[2].plot(self.rollout[8, :], label = 'Yaw Predicted')
        axs[2].legend()
        plt.suptitle(self.title)
        plt.savefig('rpy.png')

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(self.data[9, :], label = 'Roll Rate Actual')
        axs[0].plot(self.rollout[9, :], label = 'Roll Rate Predicted')
        axs[0].legend()

        axs[1].plot(self.data[10, :], label = 'Pitch Rate Actual')
        axs[1].plot(self.rollout[10, :], label = 'Pitch Rate Predicted')
        axs[1].legend()

        axs[2].plot(self.data[11, :], label = 'Yaw Rate Actual')
        axs[2].plot(self.rollout[11, :], label = 'Yaw Rate Predicted')
        axs[2].legend()
        plt.suptitle(self.title)
        plt.savefig('rpy_rates.png')
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
testNN = SampleLearnedModel(config) 
flight_file = "./bootstrap/datasets/dyn/AGGRO_000/sim_data/save-flight-04.19.2023_22.12.05.npy"
#flight_file = "test_data_dyn2.npy"
test_data = np.load(flight_file)
test_state = test_data['states'][0]

#Run Tester Class
#AnalyticalTester = TestDynamics(testAnalytical, test_state, "Analytical Model")
#AnalyticalTester.runModelStep(printout=False)
# AnalyticalTester.linear_absolute_error()
# AnalyticalTester.rotational_absolute_error()

#AnalyticalTester.runModelRollout()
#AnalyticalTester.compareTraj()

NNTester = TestDynamics(testAnalytical, testNN, test_state, "NN Model")
#NNTester.runModelStep(printout=False)
#NNTester.linear_absolute_error()
#NNTester.rotational_absolute_error()

NNTester.runModelRollout()
NNTester.compareTraj()

"""DOESNT WORK FOR PYB DATA YET (where explicit = False)"""
# #Create dyanmics model object
# testAnalyticalPYB = AnalyticalModel(config, explicit = False) 
# test_data = np.load("test_data_new.npy")
# test_state = test_data['states'][4]

# #Run Tester Class
# AnalyticalTesterPYB = TestDynamics(testAnalyticalPYB, test_state)
# AnalyticalTesterPYB.runModel(printout=True)
# AnalyticalTesterPYB.linear_absolute_error()
# AnalyticalTesterPYB.rotational_absolute_error()

print("end")

