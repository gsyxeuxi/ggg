import argparse
import sklearn
import cv2 as cv
import os
import random
import ADS1263
import pypylon.pylon as py
import Jetson.GPIO as GPIO
from multiprocessing.sharedctypes import Array, Value
import math
import csv
import time
import datetime
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import ballPlateEnv_v3
##################### limit GPU memory usage  ####################
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.set_logical_device_configuration(
        #gpus[0],
        #[tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
logical_gpus = tf.config.list_logical_devices('GPU')
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
from multiprocessing import Process
from circles_det import detect_circles_cpu
from coord_trans import coordinate_transform


Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################
# choose env
ENV_ID = 'BOP'  # environment id
RANDOM_SEED = 2  # random seed
RENDER = False  # render while training

# RL training
ALG_NAME = 'TD3'
TRAIN_EPISODES = 100  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 50  # maximum number of steps for one episode
BATCH_SIZE = 128  # update batch size
EXPLORE_STEPS = 500  # 500 for random action sampling in the beginning of training

HIDDEN_DIM = 128  # size of hidden layers for networks
UPDATE_ITR = 3  # repeated updates for single step
Q_LR = 3e-4  # q_net learning rate
POLICY_LR = 3e-4  # policy_net learning rate
POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed steps for updating the policy network and target networks
EXPLORE_NOISE_SCALE = 0.1  # range of action noise for exploration
EVAL_NOISE_SCALE = 0.2  # range of action noise for evaluation of action value
REWARD_SCALE = 1.  # value range of reward
REPLAY_BUFFER_SIZE = 1e5  # size of replay buffer

MAX_ACTION = 1
ACTION_FACT = 0.05 # -0.1 ~ 0


def PIDPlate(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y, IS_RESET):
    REF = 5.03 
    angle = [0.0, 0.0]
    angle_diff = [0.0, 0.0]
    angle_diff_sum = [0.0, 0.0]
    angle_diff_last = [0.0, 0.0] 
    angle_set = [0.0, 0.0]
    action_set_clip = [0.0, 0.0]
    kp = 0.3
    ki = 0.07
    kd = 2.0
    KEEP = False
    # set up PWM
    GPIO.setmode(GPIO.BCM)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(12, GPIO.OUT, initial=GPIO.HIGH)
    p1 = GPIO.PWM(12, 1000)
    GPIO.setup(13, GPIO.OUT, initial=GPIO.HIGH)
    p2 = GPIO.PWM(13, 1000)
    val = [0, 0]
    p1.start(val[0])
    p2.start(val[1])
    try:
        # set up ADC
        ADC = ADS1263.ADS1263()
        # choose the rate here (100Hz)
        if (ADC.ADS1263_init_ADC1('ADS1263_100SPS') == -1):
            exit()
        ADC.ADS1263_SetMode(0) # 0 is singleChannel, 1 is diffChannel
        channelList = [0, 1]  # The channel must be less than 10
        while(1):
            action_set_clip = np.clip([action_set[0], action_set[1]], -1, 1)
            if IS_RESET.value:
                if KEEP == False:
                    angle_set = 6*(np.random.rand(2)-0.5)
                    print('angle_set is', angle_set)
                    KEEP = True
                # time.sleep(3)
            else:
                #print('action is', ACTION_FACT*((action_set_clip[0]-MAX_ACTION)))
                angle_set[0] = round(ACTION_FACT*((action_set_clip[0]-MAX_ACTION) * (pos_set_x.value - real_pos_x.value) + (action_set_clip[1]-MAX_ACTION) * (vel_set_x.value - vel_x.value)), 3)
                angle_set[1] = round(ACTION_FACT*((action_set_clip[0]-MAX_ACTION) * (pos_set_y.value - real_pos_y.value) + (action_set_clip[1]-MAX_ACTION) * (vel_set_y.value - vel_y.value)), 3)
                angle_set = np.clip([angle_set[0],  angle_set[1]], -6, 6)
                KEEP = False
                # print(angle_set)
            ADC_Value = ADC.ADS1263_GetAll(channelList)    # get ADC1 value
            for i in channelList:
                if(ADC_Value[i]>>31 ==1): #received negativ value, but potentiometer should not return negativ value
                    print('negativ potentiometer value received')
                    exit()                  # p2.ChangeDutyCycle(100) 
                else:       #potentiometer received positiv value
                    #change receive data in V to angle in °
                    receive_data = ADC_Value[i] * REF / 0x7fffffff
                    angle[i] = float('%.2f' %((receive_data - 2.51) * 2.91))   # 32bit
                    # print('angle', str(i+1), ' = ', angle[i], '°', sep="")
                angle_diff[i] = angle_set[i] - angle[i]
                angle_diff_sum[i] += angle_diff[i]
                val[i] = 100 - 20 * (2.5 + kp * angle_diff[i] + ki * angle_diff_sum[i] + kd * (angle_diff[i] - angle_diff_last[i]))
                if val[i] > 100:
                    val[i] = 100
                if val[i] < 0:
                    val[i] = 0
                angle_diff_last[i] = angle_diff[i]
                if i == 0:
                    p1.ChangeDutyCycle(val[i])
                else:
                    p2.ChangeDutyCycle(val[i])
    except IOError as e:
        print(e)
    
    except KeyboardInterrupt:
        print("ctrl + c:")
        print("Program end")
        p1.stop()
        p2.stop()
        ADC.ADS1263_Exit()
        exit()

def DetectBall(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y):
    inver_matrix, transform_matrix = coordinate_transform()
    # could move in while loop later
    pos_last_x = 0
    pos_last_y = 0
    latency = 1000/60
    tlf = py.TlFactory.GetInstance()
    device = tlf.CreateFirstDevice()
    cam = py.InstantCamera(device)
    cam.Open()
    #reset the camera
    cam.UserSetSelector = "UserSet2"
    cam.UserSetLoad.Execute()
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(60)
    cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    previous_time = time.time()
    while cam.IsGrabbing():
        grabResult = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        # print(str('Number of skipped images:'), grabResult.GetNumberOfSkippedImages())
        if grabResult.GrabSucceeded():
            pos_set_trans = np.round(np.dot(transform_matrix, np.array(([pos_set_x.value*540/400],[pos_set_y.value*540/400],[1]))))
            pos_set_trans_x = int(pos_set_trans[0][0])
            pos_set_trans_y = int(pos_set_trans[1][0])
            img = grabResult.Array
            img = cv.GaussianBlur(img,(3,3),0)
            dectect_back = detect_circles_cpu(img, cv.HOUGH_GRADIENT, dp=1, min_dist=50, param1=100, param2=36, min_Radius=26, max_Radius=32)
            # img= cv.drawMarker(img, (int(pos_set_x.value), int(pos_set_y.value)), (0, 0, 255), markerType=1)
            img= cv.drawMarker(img, (int(pos_set_trans_x), int(pos_set_trans_y)), (0, 0, 255), markerType=1)
            x = dectect_back[1][0]
            y = dectect_back[1][1]
            #coordinate transform
            real_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
            real_pos_x.value = real_pos[0][0] - 1
            real_pos_y.value = real_pos[1][0]
            vel_x.value = np.round((real_pos_x.value - pos_last_x) * 1000 / latency, 3) #dt = 1/60
            vel_y.value = np.round((real_pos_y.value- pos_last_y) * 1000 / latency, 3)
            pos_last_x = real_pos_x.value
            pos_last_y = real_pos_y.value
            current_time = time.time()
            latency = round(1000 * (current_time - previous_time), 2)
            previous_time = current_time
            # print(str('latency is:'), latency, str('ms'))
            cv.namedWindow('title', cv.WINDOW_NORMAL)
            cv.imshow('title', img)
            k = cv.waitKey(1)
            if k == 27:
                break
        grabResult.Release()
    cam.StopGrabbing()
    cv.destroyAllWindows()
    cam.Close()

def trajectory(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,): 
    #l: Half of the length of the diagonal of the square
    #p: Time peroide
    l = 100
    p = 16
    a = l*np.sqrt(2)/(p/8)**2 #acceleration
    while True:
        t = time.time() % p  # Ensure that the trajectory repeats every p seconds
        if 0 <= t < p/8:
            pos_set_x.value = l - 0.5 * a * t**2/np.sqrt(2)
            pos_set_y.value = 0.5 * a * t**2/np.sqrt(2)
            vel_set_x.value = - a * t * 2/np.sqrt(2)
            vel_set_y.value = a * t * 2/np.sqrt(2)
        elif p/8 <= t < p/4:
            pos_set_x.value = 0.5 * a * (p/4-t)**2/np.sqrt(2)
            pos_set_y.value = l - 0.5 * a * (p/4-t)**2/np.sqrt(2)
            vel_set_x.value = - a * (p/4-t) * 2/np.sqrt(2)
            vel_set_y.value = a * (p/4-t) * 2/np.sqrt(2)
        elif p/4 <= t < 3*p/8:
            pos_set_x.value = -0.5 * a * (t-p/4)**2/np.sqrt(2)
            pos_set_y.value = l - 0.5 * a * (t-p/4)**2/np.sqrt(2)
            vel_set_x.value = - a * (t-p/4) * 2/np.sqrt(2)
            vel_set_y.value = - a * (t-p/4) * 2/np.sqrt(2)
        elif 3*p/8 <= t < p/2:
            pos_set_x.value = -l + 0.5 * a * (p/2-t)**2/np.sqrt(2)
            pos_set_y.value = 0.5 * a * (p/2-t)**2/np.sqrt(2)
            vel_set_x.value = - a * (p/2-t) * 2/np.sqrt(2)
            vel_set_y.value = - a * (p/2-t) * 2/np.sqrt(2)
        elif p/2 <= t < 5*p/8:
            pos_set_x.value = -l + 0.5 * a * (t-p/2)**2/np.sqrt(2)
            pos_set_y.value = -0.5 * a * (t-p/2)**2/np.sqrt(2)
            vel_set_x.value = a * (t-p/2) * 2/np.sqrt(2)
            vel_set_y.value = -a * (t-p/2) * 2/np.sqrt(2)
        elif 5*p/8 <= t < 3*p/4:
            pos_set_x.value = -0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            pos_set_y.value = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            vel_set_x.value = a * (3*p/4-t) * 2/np.sqrt(2)
            vel_set_y.value = -a * (3*p/4-t) * 2/np.sqrt(2)
        elif 3*p/4 <= t < 7*p/8:
            pos_set_x.value = 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            pos_set_y.value = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            vel_set_x.value = a * (t-3*p/4) * 2/np.sqrt(2)
            vel_set_y.value = a * (t-3*p/4) * 2/np.sqrt(2)
        else:
            pos_set_x.value = l - 0.5 * a * (t-p)**2/np.sqrt(2)
            pos_set_y.value = -0.5 * a * (t-p)**2/np.sqrt(2)
            vel_set_x.value = a * (p-t) * 2/np.sqrt(2)
            vel_set_y.value = a * (p-t) * 2/np.sqrt(2)

class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.output_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w),
            in_channels=hidden_dim, name='policy_output'
        )
        self.action_range = action_range
        self.num_actions = num_actions
        self.update_cnt = 0
        # create tensorboard logs
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('logs/td3_zupa/'): #Zustandreglerparameter
            os.makedirs('logs/td3_zupa/')
        self.log_dir = 'logs/td3_zupa/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        output = tf.nn.tanh(self.output_linear(x))  # unit range output [-1, 1]
        return output

    def evaluate(self, state, eval_noise_scale):
        """ 
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        """
        state = state.astype(np.float32)
        action = self.forward(state)

        action = self.action_range * action

        # add noise
        normal = Normal(0, 1)
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        action = action + noise
        return action

    def get_action(self, state, explore_noise_scale, greedy=False):
        """ generate action with state for interaction with envronment """
        self.update_cnt += 1
        action = self.forward([state])
        action = self.action_range * action.numpy()[0]
        if greedy:
            return action
        # add noise
        normal = Normal(0, 1)
        noise = normal.sample(action.shape) * explore_noise_scale
        action += noise
        with self.summary_writer.as_default():
                    tf.summary.scalar('c0', action[0], step=self.update_cnt)
                    tf.summary.scalar('c1', action[1], step=self.update_cnt)
        return action.numpy()

    def sample_action(self):
        """ generate random actions for exploration """
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()


class TD3:

    def __init__(
            self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, policy_target_update_interval=1,
            q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        # set train mode
        self.q_net1.train()
        self.q_net2.train()
        self.target_q_net1.eval()
        self.target_q_net2.eval()
        self.policy_net.train()
        self.target_policy_net.eval()

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)
        # create tensorboard logs
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('logs/td3/'):
            os.makedirs('logs/td3/')
        self.log_dir = 'logs/td3/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):#tau = 0.005
        """ update all networks in TD3 """
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # clipped normal noise
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
            with self.summary_writer.as_default():
                    tf.summary.scalar('q_value_loss1', q_value_loss1, step=self.update_cnt)
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
            with self.summary_writer.as_default():
                    tf.summary.scalar('q_value_loss2', q_value_loss2, step=self.update_cnt)
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            with tf.GradientTape() as p_tape:
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients
                new_q_input = tf.concat([state, new_action], 1)
                # """ implementation 1 """
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                """ implementation 2 """
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
                with self.summary_writer.as_default():
                    tf.summary.scalar('policy loss', policy_loss, step=self.update_cnt)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.q_net1.trainable_weights, extend_path('model_q_net1.npz'))
        tl.files.save_npz(self.q_net2.trainable_weights, extend_path('model_q_net2.npz'))
        tl.files.save_npz(self.target_q_net1.trainable_weights, extend_path('model_target_q_net1.npz'))
        tl.files.save_npz(self.target_q_net2.trainable_weights, extend_path('model_target_q_net2.npz'))
        tl.files.save_npz(self.policy_net.trainable_weights, extend_path('model_policy_net.npz'))
        tl.files.save_npz(self.target_policy_net.trainable_weights, extend_path('model_target_policy_net.npz'))

    def load(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        #path = os.path.join('model', '3_0.5_0.2')
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net1.npz'), self.q_net1)
        tl.files.load_and_assign_npz(extend_path('model_q_net2.npz'), self.q_net2)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net1.npz'), self.target_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net2.npz'), self.target_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_policy_net.npz'), self.policy_net)
        tl.files.load_and_assign_npz(extend_path('model_target_policy_net.npz'), self.target_policy_net)

if __name__ == '__main__':
    pos_set_x = Value('d', 70.0)
    pos_set_y = Value('d', 100.0)
    vel_set_x = Value('d', 0.0)
    vel_set_y = Value('d', 0.0)
    action_set = Array('d', [0.0, 0.0])
    real_pos_x = Value('d', 0.0)
    real_pos_y = Value('d', 0.0)
    vel_x = Value('d', 0.0)
    vel_y = Value('d', 0.0)
    IS_RESET = Value('i', 0)
    plate_process = Process(target=PIDPlate, args=(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y, IS_RESET,))
    detect_process = Process(target=DetectBall, args=(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,))
    # trajectory_process = Process(target=trajectory, args=(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,))
    plate_process.start()
    detect_process.start()
    # trajectory_process.start()
    
    #Main process training 
    arr = [real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y]
    env = ballPlateEnv_v3.Ball_On_Plate_Robot_Env(position=arr)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high  # scale action, [-action_range, action_range]

    env.reset()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    # initialization of trainer
    agent = TD3(
        state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, POLICY_TARGET_UPDATE_INTERVAL, Q_LR, POLICY_LR
    )
    t0 = time.time()

    if args.train:
        frame_idx = 0
        all_episode_reward = []
        c_value = []

        # need an extra call here to make inside functions be able to use model.forward
        # state = env.reset().astype(np.float32)
        state, _= env.reset()
        state = state.astype(np.float32)
        agent.policy_net([state])
        agent.target_policy_net([state])
        print('learn begin')
        for episode in range(TRAIN_EPISODES):
            # state = env.reset().astype(np.float32)
            print("********************")
            IS_RESET.value = 1
            state, _ = env.reset()
            print(state)
            IS_RESET.value = 0
            state = state.astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)
                    # print('action is', action)
                else:
                    action = agent.policy_net.sample_action()
                for i in range(2):
                    action_set[i] = action[i]
                if len(replay_buffer) > BATCH_SIZE:
                    for i in range(UPDATE_ITR):
                        agent.update(BATCH_SIZE, EVAL_NOISE_SCALE, REWARD_SCALE)

                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done is True else 0

                replay_buffer.push(state, action, reward, next_state, done)
                # c_value.append([action_set[0],action_set[1]])
                state = next_state
                episode_reward += reward
                frame_idx += 1

                if done:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward,
                    time.time() - t0
                )
            )

        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
        plt.figure() #create new figure
        plt.plot([i[0] for i in c_value], label='c0')
        plt.plot([i[1] for i in c_value], label='c1')
        plt.xlabel('Timestep')
        plt.ylabel('Control parameters')
        plt.legend()
        plt.savefig(os.path.join('image', '_'.join(["1", ENV_ID])))
        print('training finished, please press "ctrl+c"')
        plate_process.join()
        detect_process.join()
        # trajectory_process.join()
    
    if args.test:
        MAX_STEPS = 5000
        agent.load()
        # need an extra call here to make inside functions be able to use model.forward
        state, _= env.reset()
        state = state.astype(np.float32)
        agent.policy_net([state])

        for episode in range(TEST_EPISODES):
            print("********************")
            IS_RESET.value = True
            state, _ = env.reset()
            state = state.astype(np.float32)
            print(state)
            IS_RESET.value = False
            episode_reward = 0
            for step in range(MAX_STEPS):
                action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE, greedy=True)
                for i in range(2):
                    action_set[i] = action[i]
                print(action_set[0], action_set[1])
                time.sleep(0.3)
                state, reward, done, _= env.step(action)
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
        print('testing finished, please press "ctrl+c"')
        plate_process.join()
        detect_process.join()
        # trajectory_process.join()
