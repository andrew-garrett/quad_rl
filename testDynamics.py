from mppi.MPPI_Node import get_mppi_config
from mppi.dynamics_models import AnalyticalModel
from matplotlib import pyplot as plt

import numpy as np

class TestDynamics():
    """Class for testing performance of dynamics models"""

    def __init__(self, model, test_states):
        #Dynamics model object, to be tested
        self.model = model
        #Data from a single drone
        self.data = test_states
        _ , self.n = np.shape(test_states)

    def runModel(self, printout = False): 
        #preallocate prediction
        self.s_pred = np.zeros((12, self.n-1))
        #Simulate model for each time step
        for i in range(1, self.n):
            s_in = self.data[:12, i-1] #previous state 
            u_in = self.data[12:, i] #control action taken 
            s_out = self.data[:12, i]
            #Call model to get prediction
            self.s_pred[:, i-1] = self.model(np.reshape(s_in, (1,12)), np.reshape(u_in, (1,4)))

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


    


#TEST ANALYTICAL DYNAMICS MODEL
#Create config
config = get_mppi_config()

#Create dyanmics model object
testAnalytical = AnalyticalModel(config) 
test_data = np.load("test_data_dyn2.npy")
test_state = test_data['states'][0]

#Run Tester Class
AnalyticalTester = TestDynamics(testAnalytical, test_state)
AnalyticalTester.runModel(printout=True)
AnalyticalTester.linear_absolute_error()
AnalyticalTester.rotational_absolute_error()

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

