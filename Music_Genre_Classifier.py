from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random 
import operator
import math
import numpy as np

def distance(instance1 , instance2):
    # Returns the distance between wav files 
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    return distance

def get_nearest_k_neighbours(trainingSet, instance, k):
    """
    Return: k nearest neighbours to the instance from the trainingSet
    """
    distances = []  # Distances for each song in the trainingset to the instance
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance) + distance(instance, trainingSet[x])
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))  # Sort distances
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

def class_prediction(neighbours):
    """
    input: k closest neighbours to our instance
    output: class type with the most votes
    """
    class_votes = {}
    for x in range(len(neighbours)):
        response = neighbours[x]
        if response in class_votes:
            class_votes[response]+=1 
        else:
            class_votes[response]=1
    sorter = sorted(class_votes.items(), key = operator.itemgetter(1), reverse=True) # Sort in descending order
    return sorter[0][0]

def model_accuracy_on_testset(testSet, predictions):
    """
    input: testSet and corresponding predictions 
    output: Score (model accuracy) as a percentage 
    """
    num_correct_predictions = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            num_correct_predictions+=1

    return 100*num_correct_predictions/len(testSet)

def load_data(filename , split , trainSet , testSet):
    # Split each file in filename into trainSet/testSet with probability split/1-split
    dataset = []
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  
    for x in range(len(dataset)):
        if random.random() <split :      
            trainSet.append(dataset[x])
        else:
            testSet.append(dataset[x])  


loops = 50
total_tested=0
total_right=0
for _ in range(loops):
    # Extract features from the dataset and dump these features into a binary .dat file “my.dat”
    directory = "C:/Users/danie/source/repos/Music_Genre_Classifier/music_speech/music_wav"  # Directory to where the folders of genres are stored
    f= open("my.dat" ,'wb')     # Open for writing in binary mode
    i=0
    for folder in os.listdir(directory):
        i+=1
        if i==4:    # First 3 folders only
            break
        for file in os.listdir(directory+"/"+folder):
            (rate,sig) = wav.read(directory+"/"+folder+"/"+file)
            mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix , covariance , i)
            pickle.dump(feature , f)    # Dump feature in f ("my.dat")
    f.close()

    # Train test split
    training_set = []
    test_set = []
    load_data("my.dat" , 0.8,training_set, test_set)    # load data into training and test sets with approximately 2/3 split

    # Make prediction with KNN classifier on each song in the testSet
    num_test_songs = len(test_set)
    print("Num test songs:"+ str(num_test_songs))
    predictions = []    # Store predictions here
    for x in range(num_test_songs):
        predictions.append(class_prediction(get_nearest_k_neighbours(training_set ,test_set[x] , 1))) 
    
    score = model_accuracy_on_testset(test_set, predictions)
    print("Score:"+str(score))
    total_right+=score*num_test_songs
    total_tested+=num_test_songs

print("-------------")
print(total_right/total_tested)
print(total_tested)
print("-------------")