from mppi.MPPI_Node import get_mppi_config
from mppi.dynamics_models import AnalyticalModel

import numpy as np

#TEST ANALYTICAL DYNAMICS MODEL
#Create config
config = get_mppi_config()

#Create dyanmics model object
testAnalytical = AnalyticalModel(config) 


#Simple test case, start at zero, try to hover
state = np.zeros(12)
u = np.ones(4)*config.CF2X.HOVER_RPM
test_state_1 = testAnalytical(state, u)

#Simple test case, start at zero, try to fly upwards
state = np.zeros(12)
u = np.ones(4)*config.CF2X.HOVER_RPM*2
test_state_2 = testAnalytical(state, u)



print("end")

