import numpy as np
import matplotlib.pyplot as plt
import math
import gymnasium as gym
import sys
sys.path.append('../../DL/')
from framework import (InputLayer, FullyConnectedLayer,ReLULayer)

#create game and get info about it
ENV_NAME = 'CartPole-v1'
#env = gym.make(ENV_NAME, render_mode = 'human')  #use this to visualize game play
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n  #number of posible actions
nb_observations = env.observation_space.shape[0]  #number of features per observation

#assemble networks
#create your Main and Target networks
#Suggested Architecture:  FC(w/64 outs)->ReLu->FC(w/nb_action outs)
#TODO
#main network
#L1 = InputLayer(env)
L2 = FullyConnectedLayer(nb_observations, 64)
L3 = ReLULayer()
L4 = FullyConnectedLayer(64, nb_actions)
Layers = [L2,L3,L4]

#target network
#L5 = InputLayer(env)
L6 = FullyConnectedLayer(nb_observations, 64)
L7 = ReLULayer()
L8 = FullyConnectedLayer(64, nb_actions)
Layers_target = [L6,L7,L8]
# Xavier initial
L4.setWeights(np.random.randn(64,nb_actions) * np.sqrt(2./64))
L2.setWeights(np.random.randn(nb_observations,64) * np.sqrt(2./nb_observations))
L8.setWeights(np.random.randn(64,nb_actions) * np.sqrt(2./64))
L6.setWeights(np.random.randn(nb_observations,64) * np.sqrt(2./nb_observations))
L2.setBiases(np.zeros(64))
L4.setBiases(np.zeros(nb_actions))
L6.setBiases(np.zeros(64))
L8.setBiases(np.zeros(nb_actions))
#HYPERPARAMETERS
LEARNING_RATE = 5e-3   #learning rate
LEARNING_RATE_DECAY = 0.99999

EXPLORATION_RATE = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_RATE_DECAY = 0.98

DISCOUNT = 0.975

UPDATE_TARGET_EVERY = 100

BATCH_SIZE = 128

EPISODES = 100


results = []  #stores episode lengths


#for the history
X1 = []
X2 = []
Actions = []
Rewards = []
Terminateds = []
exit_training = False
iteration = 0

for episode in range(EPISODES):
    eplen = 0
    if exit_training:
        break
    print("Begin Episode:",episode+1)
    episode += 1
    #start the episode!
    observation, info = env.reset()
    currentObservation = np.array([observation])
    
    terminated = False
    while(not terminated):  #run the episode until it termiantes
        eplen += 1
        iteration += 1
        
        #Copy main network's weights to target one periodically
        if(iteration%UPDATE_TARGET_EVERY==0):
            #TODO
            #copy weights from main network to target network
            for L in range(len(Layers)):
                if isinstance(Layers[L], FullyConnectedLayer):
                    Layers_target[L].setWeights(Layers[L].getWeights())
                    Layers_target[L].setBiases(Layers[L].getBiases())
        
        #Go through Main network to get q-values for each action for the current observation
        #Final output is q1
        #TODO
        q1 = currentObservation.copy()
        for L in range(len(Layers)):
            q1 = Layers[L].forward(q1)
        
        
        #Choose action
        action = np.argmax(q1)  #q1 was output from main network
        
        #random choice?
        if(np.random.rand() <= EXPLORATION_RATE):
            action = np.random.choice(nb_actions)
        
        #Perform action, get reward and new state/observation
        observation, reward, terminated, truncated, info = env.step(action)
        newObservation = np.array([observation])
        
        if terminated:
            reward = 0.  #no reward if terminated
            results.append(eplen)  #add episode length to results
        
        #add to memory
        if(len(X1)==0):
            X1 = np.array(currentObservation.copy())
            X2 = np.array(newObservation.copy())
            Actions = np.array(action)
            Rewards = np.array(reward)
            Terminateds = np.array(terminated)
        else:
            X1 = np.vstack((X1,currentObservation.copy()))
            X2 = np.vstack((X2,newObservation.copy()))
            Actions = np.vstack((Actions,action))
            Rewards = np.vstack((Rewards,reward))
            Terminateds = np.vstack((Terminateds,terminated))
        
        currentObservation = newObservation.copy()
        
        
        #TRAIN!
        if(len(X1) > BATCH_SIZE):
            
            #Grab a random batch
            locs = np.random.permutation(X1.shape[0])
            locs = locs[:BATCH_SIZE]
            batchX1 = X1[locs] #current states
            batchActions = Actions[locs] #actions taken
            batchRewards = Rewards[locs] #rewards
            batchX2 = X2[locs] #next states
            batchDones = Terminateds[locs] #done status
                        
            #compute Q value using Main network
            #basically, just forward propagate (again).
            #Output is q_values.  Probably want to make a copy as q_values_orig
            #Todo
            for L in Layers:
                batchX1 = L.forward(batchX1)
                
            q_values = batchX1.copy()
            q_values_orig = batchX1.copy()
            

            
            
            #compute next Q values using Target network
            #Basically, just forward propagate next state through target network
            #Output is target_q_values
            #TODO
            for L in range(len(Layers_target)):
                batchX2 = Layers_target[L].forward(batchX2)
            target_q_values = batchX2.copy()
            
            
            #Compute target values using Bellman equation            
            max_target_q_values = np.array([np.max(target_q_values,axis=1)]).T
            targets = batchRewards + (1.-batchDones) * DISCOUNT * max_target_q_values
            
            # For learning: Adjust Q values of taken actions to match the computed targets
            # That way, we have zero loss everywhere EXCEPT where we took the action
            for i in range(len(batchActions)):
                q_values[i,batchActions[i][0]] = targets[i][0]
            
            loss = (q_values - q_values_orig)  #q_values_orig was a copy of q_values before changing values.
            grad = -2*loss
            #Backpropagate loss gradient through Main network to update its weights
            #TODO
            
            for L in reversed(range(len(Layers))):
                if isinstance(Layers[L], FullyConnectedLayer):
                    Layers[L].updateWeights(grad,LEARNING_RATE)
                grad = Layers[L].backward(grad)
                grad = np.clip(grad, -1e7, 1e7)
                if grad is None or np.isnan(grad).any():
                    exit_training = True ## kept breaking if learning rate was too high
                    break
            


            #Decay the exploration and learning rates
            EXPLORATION_RATE = np.max([EXPLORATION_RATE * EXPLORATION_RATE_DECAY,EXPLORATION_MIN])
            LEARNING_RATE = LEARNING_RATE * LEARNING_RATE_DECAY
          
env.close()


plt.plot(results,label='training')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.legend()
plt.show()

  