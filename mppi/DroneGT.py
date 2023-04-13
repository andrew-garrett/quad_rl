
import numpy as np


class DroneGT():
    """THIS IS A CLASS CONTAINING A DERIVED DYNAMICS MODEL OF THE DRONE TO SIMULATE BEHAVIOUR"""

    def __init__(self, drone_model):
        """
        drone_model: UDF File with parameters we can load into our class
        """

        #### OPTIONS ######

        self.DRONE_MODEL = drone_model
        self.URDF = self.DRONE_MODEL.value + ".urdf"

        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        

    def motorModel(self, w_i):
        """Relate angular velocities [rpm] of motors to motor forces [N] and toqrues [N-m] via simplified motor model"""
        #Clip RPM to be in feasible range 
        F_i = self.KF * np.clip(w_i, 0, self.MAX_RPM)**2 #[N]
        M_i = self.KM * np.clip(w_i, 0, self.MAX_RPM)**2 #[N/M]
        return F_i, M_i

    def dyanmics(self, state, u):
        """
        Given the current state and control input u, use dynamics to find the accelerations
        Two coupled second order ODES 
        """

        #Decompose state
        q, q_dot = state #TODO
        #Get forces and moments due to control 
        F, M = self.motorModel(u)
        #Coordinate transformation from drone frame to world frame 
        w_R_d = 0 #TODO

        # ---- Position, F = m*a ----
        f_g = 0 #force due to gravity 
        f_thrust = w_R_d @ np.array([0, 0, np.sum(F)]) #force due to thrust, rotated into world frame

        #NO EXTERNAL FORCES (DRAG, DOWNWASH, GROUND EFFECT ETV FOR NOT)#TODO
        F_sum = f_g + f_thrust #net force [N]
        a = F_sum/self.M #solve for accel [m/s^2] from net force 

        # ---- Orientation ------


        #TODO 

        return 

    def forwardProp(self, state, u, dt):
        """Given the current state, control input, and time interval, propogate the state kinematics"""

        #Get dynamics 

        #Apply via kinematic equations 
        # x = x_0 + v_0 dt + 0.5 a dt^2
        pass

    def _parseURDFParameters(self):
        """COPIED FROM BaseAviary CLass. For Parsing through a URDF file to get params"""
        
        """

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """

        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3



    

    