import gym
import numpy as np
import copy
import math
import random
import time
from collections import deque

def relu(mat):
    print(mat)
    print(type(mat))
    return np.multiply(mat,(mat>0))
    
def relu_derivative(mat):
    return (mat>0)*1



class NNLayer:
    
    def __init__(self, input_size, output_size, activation=None, lr = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        self.activation_function = activation
        self.lr = lr

    
    def forward(self, inputs, remember_for_backprop=True):
        
        input_with_bias = np.append(inputs,1)
        unactivated = np.dot(input_with_bias, self.weights)
        
        output = unactivated
        if self.activation_function != None:
            
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output    
        
    def update_weights(self, gradient):
        self.weights = self.weights - self.lr*gradient
        
    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        
        if self.activation_function != None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out),gradient_from_above)
            
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))), np.reshape(adjusted_mul, (1,len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)
        return delta_i
        
class RLAgent:
    
    env = None
    def __init__(self, env):
        self.env = env
        self.hidden_size = 24
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.num_hidden_layers = 2
        self.epsilon = 1.0
        self.memory = deque([],1000000)
        self.gamma = 0.95
        
        self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]
        for i in range(self.num_hidden_layers-1):
            self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu))
        self.layers.append(NNLayer(self.hidden_size+1, self.output_size))
        
    def select_action(self, observation):
        values = self.forward(np.asmatrix(observation))
        if (np.random.random() > self.epsilon):
            return np.argmax(values)
            print("exploited")
        else:
            return np.random.randint(self.env.action_space.n)
            
    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals
        
    def remember(self, done, action, observation, prev_obs):
        self.memory.append([done, action, observation, prev_obs])
        
    def experience_replay(self, update_size=20):
        if (len(self.memory) < update_size):
            return
        else: 
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs = self.memory[index]
                action_values = self.forward(prev_obs, remember_for_backprop=True)
                next_action_values = self.forward(new_obs, remember_for_backprop=False)
                experimental_values = np.copy(action_values)
                if done:
                    experimental_values[action_selected] = -1
                else:
                    experimental_values[action_selected] = 1 + self.gamma*np.max(next_action_values)
                self.backward(action_values, experimental_values)
        self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon*0.997
        for layer in self.layers:
            layer.lr = layer.lr if layer.lr < 0.0001 else layer.lr*0.99
        
    def backward(self, calculated_values, experimental_values): 
        
        delta = (calculated_values - experimental_values)
        
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
                





environment = gym.make("CartPole-v0")
policynetwork = RLAgent(environment)
targetnetwork = copy.deepcopy(policynetwork)
environment.reset()
explorationorexploitation = 0
exploitationtreshold = 0
exploitationrise = 0.0005
oldobservation = [0,0,0,0]
observation = [0,0,0,0]
experiences = []
memorysize = 10000
gamma = 0.999
whentoupdatethenetwork = 10
howmanytimesrange = 300
samplespertrainingtime = 10
time = 0
time = 0
good = False
done = False
for howmanytimes in range(0, howmanytimesrange):
    exploitationtreshold += exploitationrise
    print(howmanytimes)
    observation = environment.state
    time = 0
    exploitationtreshold = exploitationtreshold
    if howmanytimes % whentoupdatethenetwork == 0:
        targetnetwork = copy.deepcopy(policynetwork)
        print("updated the network")
    while True:
        oldobservation = observation
        time += 1
        
        action = policynetwork.select_action(oldobservation)
        print(oldobservation)
        print(policynetwork.forward(oldobservation))


        
        
        
        
        
        observation, reward, done, info = environment.step(action)

        
        policynetwork.remember(done, action, observation, oldobservation)
        
        totalloss = 0
        
        totalloss = 0
        totalloss0 = 0
        totalloss1 = 0
        totalones = 0
        totalzeros = 0





        
        
        for i in range(samplespertrainingtime):

            
            try:
                sampletobotrained = policynetwork.memory[random.randint(0, len(policynetwork.memory)-1)]
            except:
                sampletobotrained = policynetwork.memory[0]
            
            done1 = sampletobotrained[0]
            action_selected = sampletobotrained[1]
            new_obs = sampletobotrained[2]
            prev_obs = sampletobotrained[3]
            
            


            action_values = policynetwork.forward(prev_obs, remember_for_backprop=True)
            next_action_values = targetnetwork.forward(new_obs, remember_for_backprop=False)
            experimental_values = np.copy(action_values)
            if done1:
                experimental_values[action_selected] = -1
            else:
                experimental_values[action_selected] = 1 + gamma*np.max(next_action_values)

            policynetwork.backward(action_values, experimental_values)

            
            


           
            
            
            
           


            
        
        if done:
            environment.reset()
            
            time = 0
            break
        


policynetwork.epsilon = 0
print("Training complete.")
while True:
    
    
    action = policynetwork.select_action(observation)
    
   

    if action == 0:
        print("left")
    else:
        print("right")
    if action != 0 and action != 1:
        print("wtffff??")

    observation, reward, done, info =environment.step(action)
    environment.render()
    if done:
        environment.reset()
