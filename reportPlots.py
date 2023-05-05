from matplotlib import pyplot as plt
import numpy as np


#flight_file = "newAggroFile.npy"
flight_file_NN = "NN_DYN_4.npy"
flight_file_ANA = "ANA_DYN_4.npy"

test_data_NN = np.load(flight_file_NN)
test_state_NN = test_data_NN['states'][0]
test_desired_NN = test_data_NN['controls'][0]


test_data_ANA = np.load(flight_file_ANA)
test_state_ANA = test_data_ANA['states'][0]
test_desired_ANA = test_data_ANA['controls'][0]

_ , n = np.shape(test_state_NN)

t = np.arange(n)*(1/48)


# fig, axs = plt.subplots(6, 1)
# axs[0].plot(t, test_desired_NN[0, :], label = "Desired")
# axs[0].plot(t, test_state_NN[0, :], '--', label = "Tracked")
# axs[0].set_ylabel("x [m]")

# axs[1].plot(t, test_desired_NN[1, :], label = "Desired")
# axs[1].plot(t, test_state_NN[1, :], '--', label = "Tracked")
# axs[1].set_ylabel("y [m]")

# axs[2].plot(t, test_desired_NN[2, :], label = "Desired")
# axs[2].plot(t, test_state_NN[2, :], '--', label = "Tracked")
# axs[2].set_ylabel("z [m]")
# axs[2].set_ylim(2, 4)


# axs[3].plot(t, test_desired_NN[6, :], label = "Desired")
# axs[3].plot(t, test_state_NN[3, :], '--', label = "Tracked")
# axs[3].set_ylabel("Roll")
# axs[3].set_ylim(-3, 3)

# axs[4].plot(t, test_desired_NN[7, :], label = "Desired")
# axs[4].plot(t, test_state_NN[4, :], '--', label = "Tracked")
# axs[4].set_ylabel("Pitch")

# axs[5].plot(t, test_desired_NN[8, :], label = "Desired")
# axs[5].plot(t, test_state_NN[5, :], '--', label = "Tracked")
# axs[5].set_ylabel("Yaw")
# axs[5].set_ylim(-1, 1)
# axs[5].set_xlabel("t [s]")
# fig.align_ylabels(axs[:])
# axs[5].legend()


# fig2, axs2= plt.subplots(6, 1)
# axs[0].plot(t, test_desired_ANA[0, :], label = "Desired")
# axs[0].plot(t, test_state_ANA[0, :], '--', label = "Tracked")
# axs[0].set_ylabel("x [m]")

# axs[1].plot(t, test_desired_ANA[1, :], label = "Desired")
# axs[1].plot(t, test_state_ANA[1, :], '--', label = "Tracked")
# axs[1].set_ylabel("y [m]")

# axs[2].plot(t, test_desired_ANA[2, :], label = "Desired")
# axs[2].plot(t, test_state_ANA[2, :], '--', label = "Tracked")
# axs[2].set_ylabel("z [m]")
# axs[2].set_ylim(2, 4)


# axs2[3].plot(t, test_desired_ANA[6, :], label = "Desired")
# axs2[3].plot(t, test_state_ANA[3, :], '--', label = "Tracked")
# axs2[3].set_ylabel("Roll")
# axs2[3].set_ylim(-3, 3)

# axs2[4].plot(t, test_desired_ANA[7, :], label = "Desired")
# axs2[4].plot(t, test_state_ANA[4, :], '--', label = "Tracked")
# axs2[4].set_ylabel("Pitch")

# axs2[5].plot(t, test_desired_ANA[8, :], label = "Desired")
# axs2[5].plot(t, test_state_ANA[5, :], '--', label = "Tracked")
# axs2[5].set_ylabel("Yaw")
# axs2[5].set_ylim(-1, 1)
# axs2[5].set_xlabel("t [s]")

# axs2[5].legend()
# fig2.align_ylabels(axs2[:])

# plt.show()



fig, axs = plt.subplots(3, 1)
axs[0].plot(t, test_desired_NN[0, :], label = "Desired")
axs[0].plot(t, test_state_NN[0, :], '--', label = "NN Tracked")
axs[0].plot(t, test_state_ANA[0, :], '--', label = "Analytical Tracked")
axs[0].set_ylabel("x [m]")
axs[0].grid()

axs[1].plot(t, test_desired_NN[1, :], label = "Desired")
axs[1].plot(t, test_state_NN[1, :], '--', label = "NN Tracked")
axs[1].plot(t, test_state_ANA[1, :], '--', label = "Analytical Tracked")
axs[1].grid()

axs[1].set_ylabel("y [m]")

axs[2].plot(t, test_desired_NN[2, :], label = "Desired")
axs[2].plot(t, test_state_NN[2, :], '--', label = "NN Tracked")
axs[2].plot(t, test_state_ANA[2, :], '--', label = "Analytical Tracked")

axs[2].set_ylabel("z [m]")
axs[2].set_ylim(2, 4)
fig.align_ylabels(axs[:])
axs[2].legend()
axs[2].grid()

axs[2].set_xlabel("t [s]")

plt.show()

fig, axs = plt.subplots(3, 1)
axs[0].plot(t, test_desired_NN[3, :], label = "Desired")
axs[0].plot(t, test_state_NN[6, :], '--', label = "NN Tracked")
axs[0].plot(t, test_state_ANA[6, :], '--', label = "Analytical Tracked")
axs[0].set_ylabel("Roll [rad]")
axs[0].grid()

axs[1].plot(t, test_desired_NN[4, :], label = "Desired")
axs[1].plot(t, test_state_NN[7, :], '--', label = "NN Tracked")
axs[1].plot(t, test_state_ANA[7, :], '--', label = "Analytical Tracked")
axs[1].grid()

axs[1].set_ylabel("Pitch [rad]")

axs[2].plot(t, test_desired_NN[5, :], label = "Desired")
axs[2].plot(t, test_state_NN[8, :], '--', label = "NN Tracked")
axs[2].plot(t, test_state_ANA[8, :], '--', label = "Analytical Tracked")

axs[2].set_ylabel("Yaw [rad]")
fig.align_ylabels(axs[:])
axs[2].legend()
axs[2].grid()

axs[2].set_xlabel("t [s]")
plt.show()

print("done")

