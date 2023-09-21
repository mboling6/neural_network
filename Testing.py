from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

#testPenData() #testing for Q4

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

#testCarData()

#Helper Functions for Q5 and Q6 Analysis

def Q5():
    print("Question 5:\n")
    penAccuracies = []
    for i in range(5):
        penAccuracies.append(testPenData()[1])
    
    print("Pen Data\n")
    print("Max: " + str(max(penAccuracies)))
    print("Average: " + str(average(penAccuracies)))
    print("Standard Deviation: " + str(stDeviation(penAccuracies)))
    
    carAccuracies = []
    for i in range(5):
        carAccuracies.append(testCarData()[1])
    
    print("Car Data\n")
    print("Max: " + str(max(carAccuracies)))
    print("Average: " + str(average(carAccuracies)))
    print("Standard Deviation: " + str(stDeviation(carAccuracies)))


def Q6():
    print("Question 6:\n")
    penAccuracies = []
    carAccuracies = []
    for i in range(0, 41, 5): #0 to 40 inclusive
        penAccuracies_temp = []
        carAccuracies_temp = []
        for j in range(5):
            penAccuracies_temp.append(testPenData([i])[1])
            carAccuracies_temp.append(testCarData([i])[1])
        
        penAccuracies.append(penAccuracies_temp)
        carAccuracies.append(carAccuracies_temp)
    
    for m in range(len(penAccuracies)):
        print("Pen Data for Neural Network with " + str(m*5) + " Perceptrons in Hidden Layer")
        print("Max: " + str(max(penAccuracies[m])))
        print("Average: " + str(average(penAccuracies[m])))
        print("Standard Deviation: " + str(stDeviation(penAccuracies[m])))
    
    for n in range(len(carAccuracies)):
        print("Car Data for Neural Network with " + str(n*5) + " Perceptrons in Hidden Layer")
        print("Max: " + str(max(carAccuracies[n])))
        print("Average: " + str(average(carAccuracies[n])))
        print("Standard Deviation: " + str(stDeviation(carAccuracies[n])))


#Q5()
Q6()




