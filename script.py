import gym
import numpy as np
import copy
import math
import random
import time as time1

def relu(mat):
    #print(np.multiply(mat,(mat>0)))
    #print(mat)
    return mat*(mat>0)
    
def relu_derivative(mat):
    #print((mat>0)*1)
    return (mat>0)*1



class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.001
    alpha = 0

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        x = round(x, 3)
        #print(x)
        if x > 500:
        	#print("fefewfewfewfewgew")
        	return 1
        if x < -500:
        	return 0
        return (1/(1+(math.exp(-1*x))))
        #return relu(x)

    def dSigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
        #return relu_derivative(x)
    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output);
        
        for dendron in self.dendrons:
            #print(self.gradient)
            #print(str(dendron.connectedNeuron.output) + " !!")
            
            #print("huuuttia")
            dendron.dWeight = Neuron.eta * (self.gradient * dendron.connectedNeuron.output) + self.alpha * dendron.dWeight;
            dendron.weight = dendron.weight - dendron.dWeight;  #was originally adding, but changed it into subtracting
            dendron.connectedNeuron.addError(dendron.weight * self.gradient);
        self.error = 0;


class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword();

    def backPropagate(self, loss):
        for i in range(0, len(self.layers[-1])-1):
            #print(str(len(self.layers[-1])) + " !!!!!!")
            #self.layers[-1][i].setError(self.layers[-1][i].getOutput()- target[i]) was originally this
            self.layers[-1][i].setError(loss[i])
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()
        #print("ooff")

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            print(o)
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output




policynetwork = Network([4,10,2])
targetnetwork = copy.deepcopy(policynetwork)
environment = gym.make("CartPole-v0")
environment.reset()
explorationorexploitation = 0
exploitationtreshold = 0
exploitationrise = 0.005
oldobservation = [0,0,0,0]
observation = [0,0,0,0]
experiences = []
memorysize = 10000000
gamma = 0.95
whentoupdatethenetwork = 10
howmanytimesrange = 200
samplespertrainingtime = 10
time = 0
good = False
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
        #if time > 50:
        #    good = True
        #    print("OOOFFFFF")
        #    break
        policynetwork.setInput(observation)
        policynetwork.feedForword()
        explorationorexploitation = np.random.uniform()
        if explorationorexploitation > exploitationtreshold:
            action = random.randint(0,1)
            #print(str(action) + " and the random action is this")
        else:
            #print(observation)
            #print(policynetwork.getResults())
            action = np.argmax(policynetwork.getResults())
        policynetwork.setInput(observation)
        policynetwork.feedForword()

        #print(observation)
        #print(policynetwork.getResults())
        #print(action)
        
        
        

        observation, reward, done, info = environment.step(action)

        #print(str(observation.shape)+" !!!!!!!!!!!")
        #if howmanytimes > 200:
        #    environment.render()
        #environment.render()
        #print(done)
        if done:
            reward = -1

        experiences.append([oldobservation, action, reward, observation, done])
        #print([oldobservation, action, reward, observation])
        #print([oldobservation, action, reward, observation])
        #print(experiences[0])
        totalloss = 0
        
        totalloss = 0
        totalloss0 = 0
        totalloss1 = 0
        totalones = 0
        totalzeros = 0

        for i in range(samplespertrainingtime):

            #sampletobotrained = random.choice(experiences)
            try:
                sampletobotrained = experiences[random.randint(0, len(experiences)-1)]
            except:
                sampletobotrained = experiences[0]
                print("ooooooffffff")
            #print(sampletobotrained)
            if len(experiences) == 0:
                print("huutista")
            


           

            policynetwork.setInput(sampletobotrained[0])
            policynetwork.feedForword()
            qsa = policynetwork.getResults()

            qprimesa = np.copy(qsa)


            
            targetnetwork.setInput(sampletobotrained[3])
            targetnetwork.feedForword()
            qprimesa = targetnetwork.getResults()
            if sampletobotrained[4]:
            	qprimesa[sampletobotrained[1]] = -1
            	#print("get nea nead")
            else:
            	qprimesa[sampletobotrained[1]] = (float(sampletobotrained[2])+(float(gamma)*float(max(qprimesa))))
            	#print("dont get nea nead")
            loss = []
            
            """
            print("Reward: " + str(sampletobotrained[2]))
            print("Gamma: " + str(gamma))
            print("Maximum of q prime: " + str(max(qprimesa)))
            print("Q of s and a: " + str(qsa[sampletobotrained[1]]))

            print("Result loss: " + str(((float(sampletobotrained[2])+(float(gamma)*float(max(qprimesa))))-(qsa[sampletobotrained[1]]))**2))
            """
            """
            for paska in range(0, len(qprimesa)):
                loss.append(((float(sampletobotrained[2])+(float(gamma)*float(max(qprimesa))))-(qsa[paska])))
                #print(((float(sampletobotrained[2])+(float(gamma)*float(max(qprimesa))))-(qsa[sampletobotrained[1]]))**2)
                #print(((float(sampletobotrained[2])+(float(gamma)*float(max(qprimesa))))-(qsa[sampletobotrained[1]]))**2)
                totalloss += ((sampletobotrained[2]+gamma*max(qprimesa))-(qsa[paska]))


               
                if sampletobotrained[1] == 1:
                    totalloss1 +=((sampletobotrained[2]+(gamma*max(qprimesa)))-(qsa[paska]))
                    totalones += 1
                if sampletobotrained[1] == 0:
                    totalloss0 +=((sampletobotrained[2]+(gamma*max(qprimesa)))-(qsa[paska]))
                    totalzeros += 1
            """
            loss = [0,0]
            for i in range(0, len(qsa)):
                loss[i] = qsa[i]-qprimesa[i]
            #print(loss)
                
            #originally was this whole thing squared on both of these
            #print(loss)


            
            policynetwork.backPropagate(loss)

            #time.sleep(0.3)
            #print(loss)


            #print("Average total loss:" +str(totalloss/samplespertrainingtime))
        if len(experiences) > memorysize:

            experiences.pop(0)
        #print(len(experiences))

            
            
    

        
        #environment.render()
        if done:
            environment.reset()
            time = 0
            break
    #if good:
    #    break 
    
    
    

    howmanytimes = howmanytimes + 1

print("Training complete.")
while True:
    policynetwork.setInput(observation)
    policynetwork.feedForword()
    
    action = np.argmax(policynetwork.getResults())
    
   

    if action == 0:
        print("left")
    else:
        print("right")

    observation, reward, done, info =environment.step(action)
    print(observation)
    environment.render()
    if done:
        environment.reset()


"""

note to self:
the reason why the code outputs very low or very high values, is because it takes the argmax of the neural network output, so even when there is a very slight difference between the output it doesn't matter,
because it still takes the one with the most value, so it does not matter how close they are or it could be the way the loss is calculated i dunno
"""