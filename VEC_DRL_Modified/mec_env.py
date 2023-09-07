import numpy as np
from helper import *
import ipdb as pdb
import tensorflow as tf
import time
from scipy import special as sp
from scipy.constants import pi


class MecTerm(object):
    """
    MEC terminal parent class 
    """
    def __init__(self, user_config, train_config):
        self.rate = user_config['rate']
        self.dis = 0
        self.lane = user_config['lane']
        self.id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.data_buf_size = user_config['data_buf_size']
        self.t_factor = user_config['t_factor']
        self.penalty = user_config['penalty']
        self.seed = train_config['random_seed']
        self.sigma2 = train_config['sigma2']
        self.lamda = 7 # Wavelength
        self.train_config = train_config

        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        self.n_t = 1
        self.n_r = user_config['num_r']    
        self.DataBuf = 0
        
        self.SINR = 0
        self.Power = np.zeros(self.action_dim)
        self.Reward = 0
        self.State = []
        
        # some pre-defined parameters
        self.k = 1e-29
        self.t = 0.02
        self.L = 500 
        self.bandwidth = 1 # MHz
        self.velocity_lane1 = 20.0
        self.velocity_lane2 = 25.0
        self.velocity_lane3 = 30.0

        # self.channelModel = ARModel(self.n_t, self.n_r, rho=compute_rho(self) ,seed=train_config['random_seed'])
        self.channelModel = ARModel(self.n_t, self.n_r, seed=self.train_config['random_seed'])

        self.lane_velocities = {
        1: self.velocity_lane1,
        2: self.velocity_lane2,
        3: self.velocity_lane3
        }
        self.DISTANCE_MAP = {
        1: -250,
        2: -450,
        3: -500,
        4: -550
    }
    def dis_mov(self): 
        """ 
        Update the distance traveled based on the current lane
        Returns the updated distance 
        """
        self.dis += self.lane_velocities[self.lane] * self.t
        return self.dis

    def compute_rho(self):
        """
        Compute the Rician fading coefficient based on the terminal's lan 
        Returns the computed value of rho using the Bessel function of the first kind of 
        order zero to represent the Doppler effect in the channel model
        """
        width_lane = 5
        Hight_RSU = 10
        x_0 = np.array([1,0,0])  #  the reference position
        P_B = np.array([0,0,Hight_RSU])
        P_m = np.array([self.dis, width_lane*self.lane, Hight_RSU])
        # P_m:  3D vector representing the position of the MEC terminal (VU)
        # P_B: 3D vector representing the position of the base station (BS)
        self.rho = sp.j0(2 * pi * self.t * self.lane_velocities[self.lane] * np.dot(x_0, (P_B - P_m)) / (np.linalg.norm(P_B - P_m) * self.lamda))
        return self.rho


    def sampleCh(self):
        """ 
        Sample the channel based on the current distance and Rician fading coefficient
        Returns the sampled channel 
        """
        self.compute_rho()
        self.Channel = self.channelModel.sampleCh(self.dis,self.rho,self.lane)
        return self.Channel

    def getCh(self):
        """
        Get the channel without sampling
        """
        self.Channel = self.channelModel.getCh(self.dis,self.lane)    
        return self.Channel
   
    def setSINR(self, sinr):
        """
          Set the SINR value and update the channel based on the current SINR
        """
        self.SINR = sinr
        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.State = np.array([self.DataBuf, sinr, self.dis])

    def localProc(self, p):
        """"
        Perform local data processing based on the given power value
        Returns the processed data size
        """ 
        return np.power(p/self.k, 1.0/3.0)*self.t/self.L/1000 #unit:kbit
    
    def localProcRev(self, b):
        """ 
        Reverse the local data processing based on the given data size
        Returns the power required for reverse processing 
        """
        return np.power(b*1000*self.L/self.t, 3.0)*self.k
    
    def offloadRev(self, b):
        """ 
        Reverse the offloading data processing based on the given data size
        Returns the power required for reverse offloading 
        """
        return (np.power(2.0, b/(self.t*self.bandwidth*1000))-1)*self.sigma2/np.power(np.linalg.norm(self.Channel),2)
    
    def offloadRev2(self, b):
        """ 
        Reverse the offloading data processing based on the given data size
        Returns the power required for reverse offloading, limited by SINR constraint 
        """
        return self.action_bound if self.SINR <= 1e-12 else (np.power(2.0, b)-1)/self.SINR
        
    def sampleData(self):
        """ 
        Sample the data transmission based on the current power and SINR values
        Returns the transmitted data size, processed data size, received data size,
        excess power, and excess data size
        """
        data_t = np.log2(1 + self.Power[0]*self.SINR)*self.t*self.bandwidth*1000 #unit:kbits
        data_p = self.localProc(self.Power[1])
        over_power = 0
        self.DataBuf -= data_t+data_p
        
        
        if self.DataBuf < 0:
            over_power = self.Power[1] - self.localProcRev(np.fmax(0, self.DataBuf+data_p))
            self.overdata = -self.DataBuf
            self.DataBuf = 0
        else:
            self.overdata = 0
        # Génération d'une quantité aléatoire de données reçues (data_r) 
        # en utilisant la distribution de Poisson avec le taux d'arrivée (rate). 
        # Cette quantité représente les données qui arrivent au tampon.
        data_r = np.random.poisson(self.rate) #unit :mbit
        # data_r = 4  #unit：mbps
        # print('data_r:',data_r)
        self.DataBuf += data_r*self.t*1000 #unit:kbit
        # print (data_t,data_p)
        return data_t, data_p, data_r, over_power, self.overdata
    
    def buffer_reset(self, rate, seqCount):
        """ 
        Reset the data buffer with a given rate and sequence count
        Returns the new data buffer size 
        """
        self.rate = rate
        #  Le tampon de données de l'utilisateur est réinitialisé avec une valeur aléatoire
        #  comprise entre 0 et la taille maximale du tampon de données, divisée par 2.
        #  Cela simule la quantité de données déjà présente dans le tampon.
        self.DataBuf = np.random.randint(0, self.data_buf_size-1)/2.0
        # échantillonner un canal (channel) aléatoire pour l'utilisateur. 
        # pour simuler les variations dans les canaux de communication sans fil.
        self.sampleCh()
        if seqCount >= self.init_seqCnt:
            self.isUpdateActor = True
        return self.DataBuf

    def dis_reset(self):
        """ Reset the distance traveled by the terminal based on the terminal ID """
        self.dis = self.DISTANCE_MAP.get(int(self.id), 0)

    def disreset_for_test(self):
        """ Reset the distance traveled by the terminal based on the terminal ID """
        self.dis = self.DISTANCE_MAP.get(int(self.id), 0)


class MecTermLD(MecTerm):
    """
    MEC terminal class for loading from stored models
    The MecTermLD class is used for loading pre-trained models and making predictions based on those models
    """
    
    def __init__(self, sess, user_config, train_config):
        # Call the parent class (MecTerm) constructor to initialize attributes
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess  # Session
        
        # Load the pre-trained model using TensorFlow
        saver = tf.train.import_meta_graph(user_config['meta_path'])
        saver.restore(sess, user_config['model_path'])

        # Get the input and output tensors from the loaded model
        graph = tf.get_default_graph()
        input_str = "input_" + self.id + "/X:0"
        output_str = "output_" + self.id + ":0"
        self.inputs = graph.get_tensor_by_name(input_str)
        if not 'action_level' in user_config:
            self.out = graph.get_tensor_by_name(output_str)

    def feedback(self, sinr):
        """ - Provide feedback to the MEC terminal based on the given SINR value
            - Update the SINR and compute the next state
            - Sample the data transmission and calculate the reward for the current slot
            - Estimate the channel for the next slot
            - Update the system state
            ( Return relevant information including reward, power values, etc.
        """
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []
        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf

        # estimate the channel for next slot
        self.dis_mov()
        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])
        
        # update system state
        self.State = self.next_state
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata
    
    def predict(self, isRandom):
        # Make a power allocation prediction based on the current state
        # Returns the predicted power allocation and a zero-filled array
        self.Power = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        return self.Power, np.zeros(self.action_dim)
        
class MecTermDQN_LD(MecTermLD):
    """
    MEC terminal class for loading from stored models of DQN
    The MecTermDQN_LD class is specifically used for loading pre-trained models of DQN (Deep Q-Networks) 
    and making predictions based on those models.
    """
    def __init__(self, sess, user_config, train_config):
        MecTermLD.__init__(self, sess, user_config, train_config)
        graph = tf.get_default_graph()
        self.action_level = user_config['action_level']
        self.action = 0
        
        output_str = "output_" + self.id + "/BiasAdd:0"
        self.out = graph.get_tensor_by_name(output_str)
        # Create a lookup table for power allocation based on the action level
        # The table is a 2D numpy array where each row represents the power values for each action dimension.
        self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])
        
    def predict(self, isRandom):
        """ 
        Make a power allocation prediction based on the current state
        Returns the predicted power allocation and a zero-filled array
        It first runs the prediction operation using the TensorFlow session (self.sess) 
        and the input tensor (self.inputs), which gives the Q-values for each action.
        It selects the action with the highest Q-value (self.action) and uses it 
        to determine the corresponding power allocation from the lookup table (self.table).
        """
        q_out = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        self.action = np.argmax(q_out)
        action_tmp = self.action
        for i in range(self.action_dim):
            self.Power[i] = self.table[i, action_tmp % self.action_level]
            action_tmp //= self.action_level
        return self.Power, np.zeros(self.action_dim)
        
class MecTermGD(MecTerm):
    """
    MEC terminal class using Greedy algorithms
    """
    
    def __init__(self, user_config, train_config, policy):
        MecTerm.__init__(self, user_config, train_config)
        self.policy = policy #         
        self.local_proc_max_bits = self.localProc(self.action_bound) # max processed bits per slot
        
    def feedback(self, sinr):
        """ - Update the SINR and compute the next state
            - Sample the data transmission and calculate the reward for the current slot
            - Estimate the channel for the next slot
            - Update the system state
            - Return relevant information including reward, power values, etc.
        """
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []

        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()
        
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf 

        # if self.DataBuf > self.data_buf_size:
            # isOverflow = 1 
            # self.DataBuf = self.data_buf_size
        self.dis_mov()
        self.sampleCh()
        
        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])

        # update system state
        self.State = self.next_state
        # print (self.Power)
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        # return self.Reward, np.sum(self.Power), 0, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata

    def predict(self, isRandom):
        """ Make a power allocation prediction based on the current state
        Returns the predicted power allocation and a zero-filled array
        The policy attribute determines the power allocation policy to be used. 
        If the policy is 'local', the terminal performs local data processing (self.localProcDo()) first 
        and then offloading data. Otherwise, it performs offloading data first and then local data processing."""
        data = self.DataBuf
        if self.policy == 'local':
            self.offloadDo(self.localProcDo(data))
        else: 
            self.localProcDo(self.offloadDo(data))
        
        self.Power = np.fmax(0, np.fmin(self.action_bound, self.Power))
        # print ('power:',self.Power)
        return self.Power, np.zeros([self.action_dim])
    
    def localProcDo(self, data):
        """ 
        Perform local data processing based on the available data size
        Returns the remaining data size after processing 
        """
        if self.local_proc_max_bits <= data:
            #vit assigns the maximum power value (self.action_bound) to the local processing power (self.Power[1])
            # and subtracts the maximum number of processed bits from the available data size.
            self.Power[1] = self.action_bound
            data -= self.local_proc_max_bits
        else:
            # it assigns the power required for processing the available data size 
            # and sets the remaining data size to zero.
            self.Power[1] = self.localProcRev(data)
            data = 0
        return data
    
    def offloadDo(self, data):
        """ 
        Perform offloading data based on the available data size
        Returns the remaining data size after offloading
        It computes the maximum number of bits that can be offloaded (offload_max_bits) 
        based on the SINR and time slot parameters 
        """
        offload_max_bits = np.log2(1 + self.action_bound*self.SINR)*self.t*self.bandwidth*1000
        if offload_max_bits <= data:
            self.Power[0] = self.action_bound
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev(data)
            data = 0
        return data 
    
class MecTermGD_M(MecTermGD):
    """ The modification is in the computation of offload_max_bits """
    def offloadDo(self, data):
        offload_max_bits = np.log2(1+self.SINR*self.action_bound)
        if offload_max_bits <= data:
            self.Power[0] = self.action_bound
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev2(data)
            data = 0
        return data

class MecTermRL(MecTerm):
    """
    MEC terminal class using RL
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.agent = DDPGAgent(sess, user_config, train_config)
        self.prev_Power = np.zeros(self.action_dim)  # Initialize previous power allocations

        # Check if there is an initial path specified
        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config['init_seqCnt']
            self.isUpdateActor = False # the actor network should not be updated initially

    def feedback(self, sinr):
        """ 
        Update the SINR and compute the next state
        Sample the data transmission and calculate the reward for the current slot
        Estimate the channel for the next slot
        Update the system state
        Return relevant information including reward, power values, etc. 
        """
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []
        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()
        # Calculate the difference between previous and current powers
        power_difference = abs(self.Power - self.prev_Power)
        #print(power_difference)
        # Update previous powers
        self.prev_Power = np.copy(self.Power)
        # get the reward for the current slot
        # Cela pénalise la consommation d'énergie (somme de puissances) 
        # et le contenu du tampon de données.
        w_1 = self.t_factor
        w_2 = 0.1
        w_3 = (0.9-self.t_factor)
        #self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf
        self.Reward = -w_1*np.sum(self.Power)*10 - w_2*self.DataBuf + w_3*sum(power_difference)
        # self.Reward = -self.t_factor*(self.Power[1])*10 - self.Power[0]*np.log2(sinr)- (1-self.t_factor)*self.DataBuf - (1-self.t_factor)*overdata

        # estimate the channel for next slot
        self.dis_mov()
        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])
        
        # update system state
        
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        # return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata


    def predict(self, isRandom):
        """
        Make a power allocation prediction based on the current state.
        Returns the predicted power allocation and noise values
        """
        self.prev_Power = np.copy(self.Power)
        power, noise = self.agent.predict(self.State, self.isUpdateActor)
        # The power allocation values are then adjusted to be within the valid range (self.action_bound)
        self.Power = np.fmax(0, np.fmin(self.action_bound, power))
        # self.Power = [0.01,0.1]
        #The function returns the adjusted power allocation values and the noise values.
        return self.Power, noise

    def AgentUpdate(self,done):
        """ Update the RL agent based on the current state, power allocation, reward, next state, and update flag"""
        self.agent.update(self.State, self.Power, self.Reward, done, self.next_state, self.isUpdateActor)
        self.State = self.next_state
       
class MecTermDQN(MecTerm):
    """
    MEC terminal class using DQN
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.action_level = user_config['action_level']
        self.agent = DQNAgent(sess, user_config, train_config)
        self.action = 0
        # Create a lookup table for power allocation based on the action level
        # The table is a 2D numpy array where each row represents the power values for each action dimension.
        self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])


    def feedback(self, sinr):
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []
        # Update the data buffer

        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()

        # Get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf

        # Estimate the channel for next slot
        self.dis_mov()
        self.sampleCh()

        # Update the channel and state
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])
        # Update system state
        # Return the reward in this slot
        sum_power = np.sum(self.Power)
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata


    def AgentUpdate(self,done):
        """ Update the DQN agent based on the current state, action, reward, next state, and done flag"""
        self.agent.update(self.State, self.action, self.Reward, done, self.next_state)
        self.State = self.next_state
       
    def predict(self, isRandom):
        """ 
        It uses the DQN agent (self.agent) to predict the action, 
        which represents the index in the lookup table (self.table) to determine the power allocation values.
        It adjusts the action value by using the modulo operator (%) and integer division (//) 
        to extract the power values from the lookup table 
        """
        # print ('self.table:',self.table) 
        self.action, noise = self.agent.predict(self.State)
        # print ('action:',self.action)
        action_tmp = self.action
        for i in range(self.action_dim):
            # print ('action_tmp ,self.action_level',action_tmp,self.action_level)
            self.Power[i] = self.table[i, action_tmp % self.action_level]
            action_tmp //= self.action_level
        # print ( 'self.Power:',self.Power)
        return self.Power, noise

class MecSvrEnv(object):
    """
    Simulation environment
    """
    def __init__(self, user_list, Train_vehicle_ID, sigma2, max_len,mode='train'): 
        self.user_list = user_list  # store instances of the MecTermRL class for each user
        self.num_user = len(user_list)
        self.Train_vehicle_ID = Train_vehicle_ID-1
        self.sigma2 = sigma2
        self.count = 0  #  the count of transmission steps
        self.seqCount = 0
        self.max_len = max_len  #  the maximum length of an episode
        self.mode = mode  # the mode of the environment (train or test).
        # self.seed = 0

    def init_target_network(self):
        self.user_list[self.Train_vehicle_ID].agent.init_target_network()

    def step_transmit(self, isRandom=True):
        """
        - Get the channel vectors
        - Get the transmit powers
        - Compute the SINR for each user
        - Feedback SINR to each user and receive relevant information 
        """
        i = self.Train_vehicle_ID # the id of vehicle for training
        # get the channel vectors 
        # print('\ncount:',self.count)
        # print('\ndis of step %d:'%self.count,[user.dis for user in self.user_list])
        
        channels = []
        channels.append(self.user_list[i].getCh())
        for user in self.user_list:
            # Ensure whether two user are in the same BS' cover area 
            if (user.dis+250)//500 == (self.user_list[i].dis+250)//500 and int(user.id)!=self.Train_vehicle_ID+1:
                channels.append(user.getCh())

        # channels = np.transpose(channels)
        # channels = np.transpose([user.getCh() for user in self.user_list if (user.dis+50)//100 == (self.user_list[i].dis+50)//100])
        # print('channels:',np.linalg.norm(channels, axis=1))
        # get the transmit powers

        ''' 
        it uses the trained models of the MEC terminals to predict the transmit powers for the training vehicle.
        Next, it computes the Signal-to-Interference-plus-Noise Ratio (SINR) for each user using the channel vectors.
        Finally, it provides feedback to the training vehicle and receives
        the corresponding rewards, powers, noises, data transmission sizes, next channels
        '''
        powers, noises = self.user_list[i].predict(isRandom)
        # what's the difference between this powers and the other ones

        sinr_list = self.compute_sinr(channels)

        rewards = 0
        powers = 0
        over_powers = 0
        data_ts = 0
        data_ps = 0
        data_rs = 0
        data_buf_sizes = 0
        next_channels = 0
        isOverflows = 0
        power_offload = 0
        power_local = 0
        
        self.count += 1

        # print('sinr_list:',sinr_list)
        # feedback the sinr to each user
        # Add last_powers
        [rewards, powers, over_powers, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows, power_offload, power_local ,overdata] = self.user_list[i].feedback(sinr_list[0])
        # print ('self.mode:',self.mode)
        if self.mode == 'train':
            self.user_list[i].AgentUpdate(self.count >= self.max_len)

        # update the distance of other vehicle that isn't training 
        for user in self.user_list:
            if int(user.id) != self.Train_vehicle_ID + 1:  
                user.dis_mov()

        return rewards, self.count >= self.max_len, powers, over_powers, noises, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows, power_offload, power_local, sinr_list[0],overdata


    def compute_sinr(self, channels):
        """ 
        Spatial-Domain MU-MIMO ZF
        It first computes the pseudo-inverse of the transpose of the channel matrix. 
        Then, it calculates the noise power for each user by taking the norm of the pseudo-inverse.
        Finally, it computes the SINR as the inverse of the noise power.
        """
        H_inv = np.linalg.pinv(np.transpose(channels))
        noise = np.power(np.linalg.norm(H_inv, axis=1),2)*self.sigma2
        sinr_list = 1 / noise
        return sinr_list

    def reset(self, isTrain=True):
        """ 
        Reset the environment for a new episode.
        Return the initial data buffer size 
        """
        i = self.Train_vehicle_ID  # the id of vehicle for training 
        self.count = 0
        if isTrain:
            init_data_buf_size = self.user_list[i].buffer_reset(self.user_list[i].rate, self.seqCount) 
            # print('initial buffer:',self.user_list[i].DataBuf)
            # dis = [user.dis_reset() for user in self.user_list]
            for user in self.user_list:
                if self.mode == 'train':
                    user.dis_reset()
                elif self.mode == 'test':
                    user.disreset_for_test()

            # get the channel vectors   
            channels = [user.getCh() for user in self.user_list]
            # print (channels,'\n',np.linalg.norm(channels, axis=1))
            # compute the sinr for each user
            sinr_list = self.compute_sinr(channels)
            # print (sinr_list)

        else:
            init_data_buf_size = 0 
            sinr_list = [0 for user in self.user_list]

        self.user_list[i].setSINR(sinr_list[i])
            
        self.seqCount += 1
        return init_data_buf_size