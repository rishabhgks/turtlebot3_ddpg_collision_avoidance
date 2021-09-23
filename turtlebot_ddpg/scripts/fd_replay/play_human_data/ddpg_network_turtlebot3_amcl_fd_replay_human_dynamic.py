#! /usr/bin/env python

# this network can change both heading and velocity

import ddpg_turtlebot_turtlebot3_amcl_fd_replay_human_multi
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
import os.path
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import rospy
import rospkg
from priortized_replay_buffer import PrioritizedReplayBuffer
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse, DeleteModel, DeleteModelRequest, DeleteModelResponse
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
import shlex
from psutil import Popen
import roslaunch
import os
tim = time.time()
os.mkdir('ddpg_per_' + str(tim))


obstacle_loc = [[-4.04425, 3.20234], [-5.14539, 7.37162], [-8.63656, 8.3542], [8.99149, 8.22035], [8.52195, -2.66946], [-1.71052, 6.6387], \
	[-6.50286, -0.994075], [5.0846, 6.34804], [4.12434, -1.66097], [-7.26809, -9.17929], [7.65097, -9.61432], [0.166115, -8.22319], \
		[0.923208, 3.95612], [4.48937, 3.02562], [-2.26939, -2.15261], [-5.00778, -5.72124], [2.4509, -5.46721], [7.31786, -5.83205], \
			[-8.39409, 2.53726], [8.62944, 3.46707], [-8.03787, -4.33587], [-13.2814, 13.1262], [7.39829, 0.553822]] 

def batch_stack_samples(samples):
	array = np.array(samples)
	#before_current_states = np.stack(array[:,0])
	current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
	actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
	rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
	new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
	dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
	weights = np.stack(array[:,5]).reshape((array.shape[0],-1))
	indices = np.stack(array[:,6]).reshape((array.shape[0],-1))
	eps_d = np.stack(array[:,7]).reshape((array.shape[0],-1))



	return current_states, actions, rewards, new_states, dones, weights, indices, eps_d


# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.gamma = .90
		self.tau   = .01


		self.buffer_size = 1000000
		self.batch_size = 512

		self.hyper_parameters_lambda3 = 0.2
		self.hyper_parameters_eps = 0.2
		self.hyper_parameters_eps_d = 0.4

		self.demo_size = 1000

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = PrioritizedReplayBuffer() #deque(maxlen=40000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32,
			[None, self.env.action_space.shape[0]]) # where we will feed de/dC (from critic)

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output,
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output,
			self.critic_action_input) # where we calcaulte de/dC for feeding above

		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		h1 = Dense(1024, activation='relu')(state_input)
		#h2 = Dense(1000, activation='relu')(h1)
		h2 = Dense(1024, activation='relu')(h1)
		h3 = Dense(1024, activation='relu')(h2)
		delta_theta = Dense(1, activation='tanh')(h3) 
		speed = Dense(1, activation='sigmoid')(h3) # sigmoid makes the output to be range [0, 1]

		#output = Dense(self.env.action_space.shape[0], activation='tanh')(h3)
		#output = Concatenate()([delta_theta])#merge([delta_theta, speed],mode='concat')
		output = Concatenate()([delta_theta, speed])
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		state_h1 = Dense(1024, activation='relu')(state_input)
		#state_h2 = Dense(1000)(state_h1)

		action_input = Input(shape=self.env.action_space.shape)
		action_h1    = Dense(1024)(action_input)

		merged    = Concatenate()([state_h1, action_h1])
		merged_h1 = Dense(1024, activation='relu')(merged)
		merged_h2 = Dense(1024, activation='relu')(merged_h1)
		output = Dense(1, activation='linear')(merged_h2)
		model  = Model(input=[state_input,action_input], output=output)

		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		
		indice = len(self.memory.memory_data())

		target_actions = self.target_actor_model.predict(new_state)
		future_rewards = self.target_critic_model.predict([new_state, target_actions])
		rewards = reward + self.gamma* future_rewards * (1 - done)

		# get critic_loss_element_wise and actor_loss_element
		critic_values = self.critic_model.predict([cur_state, action])
		critic_loss_element = np.power((critic_values-rewards), 2)
		predicted_action = self.actor_model.predict(cur_state)
		actor_loss_element = self.critic_model.predict([cur_state, predicted_action])

		new_priorities  = critic_loss_element
		new_priorities += self.hyper_parameters_lambda3 * np.power(actor_loss_element,2)
		new_priorities += self.hyper_parameters_eps
		# new_priorities += self.hyper_parameters_eps_d

		self.memory.add(cur_state, action, reward, new_state, done, indice, new_priorities)  # add to buffer, instead of sampling batch



	def read_human_data(self):
		mat_contents = sio.loadmat('human_data.mat')
		a = mat_contents['data']
		
		for i in range(self.demo_size):
			cur_state = a[i][0:28]
			action = a[i][28:30]
			reward = a[i][30]
			new_state = a[i][31:59]
			done = a[i][59]
			cur_state = cur_state.reshape(1,28)
			action = action.reshape(1,2)
			#array_reward = np.array(reward)
			#reward = self.array_reward.reshape(1,1)
			new_state = new_state.reshape(1,28)
			indice = i
			new_priorities = 1
			action[0][1] = action[0][1]/0.26
			# print("angular velocity recorded is %s", action[0][0])
			# print("linear velocity recorded is %s", action[0][1])
			self.memory.add(cur_state, action, reward, new_state, done, indice, new_priorities)


	def _train_critic_actor(self, samples):
 
   		# 1, sample to get states, actions, rewards, new_states, dones
   		# 2, calculate weights, indices, eps_d
   		# 3, get critic_loss_element_wise
   		# 4, train critic based on weights
   		# 5, train actor based on weights
   		# 6, update target network?
   		# 7, update priorities for sampling


   		# 1, sample
		cur_states, actions, rewards, new_states, dones, weights, indices, eps_d = samples #batch_stack_samples(samples)
		target_actions = self.target_actor_model.predict(new_states)
		future_rewards = self.target_critic_model.predict([new_states, target_actions])
		rewards = rewards + self.gamma* future_rewards * (1 - dones)


		# 4, train critic based on weights
		_sample_weight = weights #(rewards/rewards).flatten()
		# print("_sample_weight is %s", _sample_weight)
		evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0, sample_weight=_sample_weight)
		# print('\nhistory dict:', evaluation.history)


		# 5, train actor based on weights
		predicted_actions = self.actor_model.predict(cur_states)
		grads = self.sess.run(self.critic_grads, feed_dict={
			self.critic_state_input:  cur_states,
			self.critic_action_input: predicted_actions
		})[0]



		#calculate grads_weight for changing the actor model weight?
		grads_weight = grads
		for i in range(0, len(grads)):
			grads_weight[i][0] = grads[i][0]*_sample_weight[i]
			grads_weight[i][1] = grads[i][1]*_sample_weight[i]
		grads = grads_weight
		self.sess.run(self.optimize, feed_dict={
			self.actor_state_input: cur_states,
			self.actor_critic_grad: grads
		})
		# print("grads*weights is %s", grads)
		


		# 3, get critic_loss_element_wise
		critic_values = self.critic_model.predict([cur_states, actions])
		critic_loss_element = np.power((critic_values-rewards), 2)

		# 7, update priorities for sampling
		actor_loss_element = self.critic_model.predict([cur_states, predicted_actions])
		# print("actor_loss_element is %s", actor_loss_element)

		new_priorities  = critic_loss_element
		new_priorities += self.hyper_parameters_lambda3 * np.power(actor_loss_element,2)
		new_priorities += self.hyper_parameters_eps
		new_priorities += self.hyper_parameters_eps_d
		# print("new_priorities is %s", new_priorities)


		######################################################################
		# update priority of sampled transitions, batch_size.
		self.memory.update_priorities(indices, new_priorities)



	def read_Q_values(self, cur_states, actions):
		critic_values = self.critic_model.predict([cur_states, actions])
		return critic_values

	def train(self):
		batch_size = self.batch_size
		if len(self.memory.memory_data()) < batch_size: #batch_size:
			return
		#samples = random.sample(self.memory.memory_data(), batch_size)    # what is deque, what is random.sample? self.mempory begins with self.memory.append
		samples = self.memory.sample(1, batch_size)
		self.samples = samples
		# print("samples is %s", samples)
		# print("samples [1] is %s", samples[1])
		print("length of memory is %s", len(self.memory.memory_data()))
		# print("samples shape is %s", samples.shape)
		self._train_critic_actor(samples)


	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
		self.target_actor_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
		self.target_critic_model.set_weights(critic_target_weights)

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):  # this function returns action, which is predicted by the model. parameter is epsilon
		#self.epsilon *= self.epsilon_decay
		self.epsilon = 0.9
		eps = self.epsilon
		action = self.actor_model.predict(cur_state)
		if np.random.random() < self.epsilon:
			action[0][0] = action[0][0] + (np.random.random()-0.5)*0.4
			action[0][1] = action[0][1] + np.random.random()*0.4
			return action, eps	
		else:
			action[0][0] = (np.random.random()-0.5)*2   # angular velocity
			action[0][1] = np.random.random()   # linear velocity
			return action, eps
		

	# ========================================================================= #
	#                              save weights                            #
	# ========================================================================= #

	def save_weight(self, num_trials, trial_len):
		self.actor_model.save_weights('actormodel' + '-' +  str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)
		self.critic_model.save_weights('criticmodel' + '-' + str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)#("criticmodel.h5", overwrite=True)

	def play(self, cur_state):
		return self.actor_model.predict(cur_state)
	
def InRectPoint(obs, loc, gap):
	x = loc[0]
	y = loc[1]
	x1 = obs[0] - gap
	y1 = obs[1] - gap
	x2 = obs[0] + gap
	y2 = obs[1] + gap
	if (x > x1 and x < x2 and
		y > y1 and y < y2) :
		return True
	else :
		return False


def my_custom_random(exclude):
	while True:
		randInt = random.randint(0, len(positions))
		return my_custom_random(exclude) if randInt in exclude else randInt 


def get_position(obstacles):
	while True:
		crash = False
		randx = random.uniform(-10, 10)
		randy = random.uniform(-10, 10)
		for obs in obstacles:
			if InRectPoint(obs, [randx, randy], 1.3):
				crash = True
				break
		if not crash:
			return [randx, randy]


def get_target_position(obstacles):
	index_list = [-1, 1]
	while True:
		crash = False
		index_x = random.choice(index_list)
		index_y = random.choice(index_list)
		randx = (np.random.random()-0.5)*5 + 12*index_x
		randy = (np.random.random()-0.5)*5 + 12*index_y
		for obs in obstacles:
			if InRectPoint(obs, [randx, randy], 1.3):
				crash = True
				break
		if not crash:
			return [randx, randy]


def spawn_targets(n, sdf, spawn_model_proxy, obstacles):
	target_obs = obstacles
	for i in range(n):
		initial_pose = Pose()
		pos = get_target_position(target_obs)
		target_obs.append(pos)
		initial_pose.position.x = pos[0]
		initial_pose.position.y = pos[1]
		initial_pose.position.z = 0.0
		orint = quaternion_from_euler(0, 1.57, 0)
		initial_pose.orientation = Quaternion(orint[0],orint[1],orint[2],orint[3])
		
		spawn_request = SpawnModelRequest()
		spawn_request.model_name = 'unit_sphere_test_' + str(i)
		spawn_request.model_xml = sdf
		spawn_request.reference_frame = 'world'
		spawn_request.initial_pose = initial_pose
		
		spawn_resp = SpawnModelResponse()
		rospy.wait_for_service('gazebo/spawn_sdf_model')
		spawn_resp = spawn_model_proxy(spawn_request)
		rospy.loginfo('Model spawn {},\n{}'.format(spawn_resp.success, spawn_resp.status_message))


def spawn_robots(n, urdf, node, num_targets):
	robot_pos = obstacle_loc
	game_state_list = []
	target_count = 0
	node_process=[]
	for i in range(n):
		rospy.set_param('/robot'+str(i)+'/tf_prefix', 'robot'+str(i)+'_tf')
	for i in range(n):
		pos = get_position(robot_pos)
		robot_pos.append(pos)
		node_process1 = Popen(shlex.split("roslaunch turtlebot_ddpg one_robot_group.launch robot_name:='robot"+ str(i) +"' init_pose:='-x " \
            + str(pos[0]) + " -y " + str(pos[1]) + " -z 0' namespace:='robot" + str(i) + "'"))
		node_process2 = Popen(shlex.split('roslaunch turtlebot_ddpg turtlebot3_laser_filter_dyn.launch num:='+str(i)))
		game_state = ddpg_turtlebot_turtlebot3_amcl_fd_replay_human_multi.GameState(node, pos[0], pos[1], 0, '/robot'+str(i), 'unit_sphere_test_' + str(target_count % num_targets))   # game_state has frame_step(action) function
		game_state_list.append(game_state)
		node_process3 = Popen(shlex.split('roslaunch turtlebot_ddpg static_transform.launch robot_name:='+str(i)))
		node_process.extend([node_process1, node_process2, node_process3])
		target_count += 1
	return game_state_list, node_process


def delete_models(n, delete_model_proxy, model='target'):
	node_process = []
	for i in range(n):
		delete_request = DeleteModelRequest()
		if model == 'target':
			delete_request.model_name = 'unit_sphere_test_' + str(i)
		else:
			delete_request.model_name = 'robot' + str(i)	
			node_process1 = Popen(shlex.split('rosnode kill /robot' + str(i) + '/robot_state_publisher'))
			node_process2 = Popen(shlex.split('rosnode kill /robot' + str(i) + '_map'))
			node_process3 = Popen(shlex.split('rosnode kill /laser_filter_' + str(i)))
			rospy.delete_param('/robot'+str(i)+'/tf_prefix')
			node_process.extend([node_process1, node_process2, node_process3])
		delete_resp = DeleteModelResponse()
		rospy.wait_for_service('gazebo/delete_model')
		delete_resp = delete_model_proxy(delete_request)
		rospy.loginfo('Model spawn {},\n{}'.format(delete_resp.success, delete_resp.status_message))
	return node_process	


# def delete_robots(n, delete_model_proxy):
# 	for i in range(n):
# 		delete_request = DeleteModelRequest()
# 		delete_request.model_name = 'robot' + str(i)
		
# 		delete_resp = DeleteModelResponse()
# 		rospy.wait_for_service('gazebo/delete_model')
# 		delete_resp = delete_model_proxy(delete_request)
# 		rospy.loginfo('Model spawn {},\n{}'.format(delete_resp.success, delete_resp.status_message))


def main():
	global tim
	sess = tf.Session()
	K.set_session(sess)

	########################################################
	node = rospy.init_node('talker', anonymous=True)

	# robot_pos = []
	game_state_list = []

	# Spawning Targets
	r = rospkg.RosPack()
	file_path = r.get_path('turtlebot_ddpg')
	file_path = file_path+'/worlds/unit_sphere/model.sdf'
	f= open(file_path,'r')
	sdf = f.read()
	f.close()
	
	# Spawning Robots
	# for i in range(20):
	# 	# node_process = Popen(shlex.split("roslaunch turtlebot_ddpg turtlebot3_empty_world.launch world_file:='/home/rishabh/genesys3_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/worlds/turtlebot3_modified_maze.world'"))
	# 	launch_file = '/home/rishabh/genesys3_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/launch/turtlebot3_empty_world.launch'
	# 	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	# 	roslaunch.configure_logging(uuid)
	# 	launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
	# 	launch.start()
	# 	rospy.loginfo("started")
	# 	time.sleep(6)
	# 	robot_range = rospy.get_param('/robot_info/num_robot')
	# 	num_robots = random.randint(2, robot_range+1)
	# 	num_targets = random.randint(1, num_robots)
	# 	urdf = rospy.get_param('/robot_description')
	# 	spawn_model_proxy = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
	# 	delete_model_proxy = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
	# 	spawn_targets(num_targets, sdf, spawn_model_proxy)
	# 	spawn_robots(num_robots, urdf, node, num_targets)

	# 	# time.sleep(2)
	# 	actor_critic_list = [ActorCritic(game_state_list[i], sess) for i in range(num_robots)]
	# 	current_state_list = []
	# 	for i in range(len(game_state_list)):
	# 		current_state_list.append(game_state_list[i].reset())
	# 	# time.sleep(2)

	# 	# delete_models(num_targets, delete_model_proxy, model='target')
	# 	# delete_models(num_robots, delete_model_proxy, model='robot')
	# 	# node_process.terminate()
	# 	launch.shutdown()
	# 	node_process2 = Popen(shlex.split('rosnode kill /gazebo'))
	# 	node_process2 = Popen(shlex.split('rosnode kill /gazebo_gui'))
	########################################################
	num_trials = 1000
	trial_len  = 2000
	train_indicator = 0

	#actor_critic.read_human_data()
	
	step_reward = [0, 0, 0, 0]
	step_Q = [0,0]
	step = 0

	if (train_indicator==2):
		for i in range(num_trials):
			print("trial:" + str(i))
			#game_state.game_step(0.3, 0.2, 0.0)
			#game_state.reset()

			current_state = game_state.reset()
			##############################################################################################
			total_reward = 0
			
			for j in range(100):
				step = step +1
				print("step is %s", step)


				###########################################################################################
				#print('wanted value is %s:', game_state.observation_space.shape[0])
				current_state = current_state.reshape((1, game_state.observation_space.shape[0]))
				action, eps = actor_critic.act(current_state)
				action = action.reshape((1, game_state.action_space.shape[0]))
				print("action is speed: %s, angular: %s", action[0][1], action[0][0])
				reward, new_state, crashed_value = game_state.game_step(0.1, action[0][1]*5, action[0][0]*5) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				

	

	if (train_indicator==1):

		# actor_critic.actor_model.load_weights("actormodel-90-1000.h5")
		# actor_critic.critic_model.load_weights("criticmodel-90-1000.h5")
		for i in range(num_trials):
			print("trial:" + str(i))
			#game_state.game_step(0.3, 0.2, 0.0)
			#game_state.reset()

			current_state = game_state.reset()
			##############################################################################################
			total_reward = 0
			
			for j in range(trial_len):

				###########################################################################################
				#print('wanted value is %s:', game_state.observation_space.shape[0])
				current_state = current_state.reshape((1, game_state.observation_space.shape[0]))
				action, eps = actor_critic.act(current_state)
				action = action.reshape((1, game_state.action_space.shape[0]))
				print("action is speed: %s, angular: %s", action[0][1], action[0][0])
				reward, new_state, crashed_value = game_state.game_step(0.1, action[0][1], action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
				total_reward = total_reward + reward
				###########################################################################################

				if j == (trial_len - 1):
					crashed_value = 1
					print("this is reward:", total_reward)
					print('eps is', eps)

				step = step + 1
				#plot_reward(step,reward,ax,fig)
				step_reward = np.append(step_reward,[step,reward])
				sio.savemat('step_reward.mat',{'data':step_reward},True,'5', False, False,'row')
				print("step is %s", step)

				Q_values = actor_critic.read_Q_values(current_state, action)
				step_Q = np.append(step_Q,[step,Q_values[0][0]])
				print("Q_values is %s", Q_values[0][0])
				sio.savemat('step_Q.mat',{'data':step_Q},True,'5', False, False,'row')

				start_time = time.time()

				if (j % 5 == 0):
					actor_critic.train()
					actor_critic.update_target()   

				end_time = time.time()
				print("train time is %s", (end_time - start_time))
				
				new_state = new_state.reshape((1, game_state.observation_space.shape[0]))

				# print shape of current_state
				#print("current_state is %s", current_state)
				##########################################################################################
				actor_critic.remember(current_state, action, reward, new_state, crashed_value)
				current_state = new_state



				##########################################################################################
			if (i % 10==0):
				actor_critic.save_weight(i, trial_len)

		

	if train_indicator==0:
		for i in range(num_trials):
			print("trial:" + str(i))
			launch_file = '/home/rishabh/genesys3_ws/src/turtlebot3_ddpg_collision_avoidance/turtlebot_ddpg/launch/turtlebot3_empty_world.launch'
			uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
			roslaunch.configure_logging(uuid)
			launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
			launch.start()
			rospy.loginfo("started")
			time.sleep(6)
			robot_range = rospy.get_param('/robot_info/num_robot')
			num_robots = random.randint(2, robot_range+1)
			num_targets = random.randint(1, num_robots-1)
			urdf = rospy.get_param('/robot_description')
			spawn_model_proxy = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
			delete_model_proxy = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
			spawn_targets(num_targets, sdf, spawn_model_proxy, obstacle_loc)
			game_state_list, robot_node_process = spawn_robots(num_robots, urdf, node, num_targets)

			# time.sleep(2)
			actor_critic_list = [ActorCritic(game_state_list[i], sess) for i in range(num_robots)]
			current_state_list = []
			for i in range(len(game_state_list)):
				current_state_list.append(game_state_list[i].reset())
			# time.sleep(2)
			for j in range(num_robots):
				current_state_list[j] = game_state_list[j].reset()
			for j in range(num_robots):
				# actor_critic.actor_model.load_weights("actormodel-160-500.h5")
				# actor_critic.critic_model.load_weights("criticmodel-160-500.h5")
				# actor_critic_list[j].actor_model.load_weights("actormodel-10-2000.h5")
				# actor_critic_list[j].critic_model.load_weights("criticmodel-10-2000.h5")
				actor_critic_list[j].actor_model.load_weights("/home/rishabh/Downloads/actormodel-500-2000.h5")
				actor_critic_list[j].critic_model.load_weights("/home/rishabh/Downloads/criticmodel-500-2000.h5")
			##############################################################################################
			total_reward_list = [0 for i in range(num_robots)]
			
			for j in range(trial_len):
				do_reset = True
				for k in range(num_robots):
					if not game_state_list[k].arrived and not game_state_list[k].crashed:
						# Robot i
						###########################################################################################
						current_state_list[k] = current_state_list[k].reshape((1, game_state_list[k].observation_space.shape[0]))

						start_time = time.time()
						action = actor_critic_list[k].play(current_state_list[k])  # need to change the network input output, do I need to change the output to be [0, 2*pi]
						action = action.reshape((1, game_state_list[k].action_space.shape[0]))
						end_time = time.time()
						print(1/(end_time - start_time), "fps for calculating step {} for robot {}".format(j, k))

						reward, new_state, crashed_value = game_state_list[k].game_step(0.1, action[0][1], action[0][0]) # we get reward and state here, then we need to calculate if it is crashed! for 'dones' value
						total_reward_list[k] = total_reward_list[k] + reward
						###########################################################################################

						if j == (trial_len - 1):
							crashed_value = 1
							print("this is reward:", total_reward_list[k])
						
						step += 1
						step_reward = np.append(step_reward,[i+1, step,reward, tim])
						sio.savemat('ddpg_per_' + str(tim) + '/iter{}_step_debug_robot{}_reward.mat'.format(i, k),{'data':step_reward},True,'5', False, False,'row')

						new_state = new_state.reshape((1, game_state_list[k].observation_space.shape[0]))
						# actor_critic.remember(cur_state, action, reward, new_state, done)   # remember all the data using memory, memory data will be samples to samples automatically.
						# cur_state = new_state

						##########################################################################################
						#actor_critic.remember(current_state, action, reward, new_state, crashed_value)
						current_state_list[k] = new_state

						##########################################################################################

					do_reset = do_reset & (game_state_list[k].arrived | game_state_list[k].crashed)
				if do_reset:
					for k in range(num_robots):
						game_state_list[k].reset()

			# delete_models(num_targets, delete_model_proxy, model='target')
			# delete_models(num_robots, delete_model_proxy, model='robot')
			# node_process.terminate()
			for i in range(robot_node_process):
				robot_node_process[i].terminate()
			launch.shutdown()
			node_process2 = Popen(shlex.split('rosnode kill /gazebo'))
			node_process2.terminate
			node_process2 = Popen(shlex.split('rosnode kill /gazebo_gui'))
			node_process2.terminate()

if __name__ == "__main__":
	main()
