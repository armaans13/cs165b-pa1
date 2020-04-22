# Starter code for CS 165B HW2 Spring 2019
import math
import numpy as np 


def centroid(data):
    centroid = np.mean(data,axis = 0)

    return centroid

def discriminantLine (centroidPos, centroidNeg):
    line = (np.linalg.norm(centroidPos) ** 2 - np.linalg.norm(centroidNeg) ** 2) / 2.0
    return line

def findOrtogonalW (centroidPos,centroidNeg):
    return np.subtract(centroidPos,centroidNeg)
    #[i - j for (i, j) in zip(centroidPos, centroidNeg)]

def classifyPoint (dataPoint, orthogAB,orthogAC,orthogBC,discrimAB,discrimAC,discrimBC):
    #print(dataPoint)
    dotAB = np.dot(dataPoint,orthogAB)
    dotAC = np.dot(dataPoint,orthogAC)
    dotBC = np.dot(dataPoint,orthogBC)

    if dotAB - discrimAB > 0:
        # between A or B, it's A. So check between A or C
        if dotAC - discrimAC > 0:
            return 1
        else:
            return 3
    else:
        # between A or B, it's B. So check between B or C
        if dotBC - discrimBC > 0:
            return 2
        else:
            return 3
    



def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """


    # TODO: IMPLEMENT
    #Define dimension and class variables
    dimension = training_input[0][0]
    sizeOfA = training_input[0][1]
    sizeOfB = training_input[0][2]
    sizeOfC = training_input[0][3]


    classOfA = training_input[1:1+sizeOfA]
    classOfB = training_input[2 + sizeOfA: 2 + sizeOfA + sizeOfB]
    classOfC = training_input[1 + sizeOfA + sizeOfB: ]

    print(len(classOfA))
    print(len(classOfB))
    print(len(classOfC))



    #compute centroids of A,B,C
    centroidOfA = centroid(classOfA)
    centroidOfB = centroid(classOfB)
    centroidOfC = centroid(classOfC)

    print(centroidOfA)
    print(centroidOfB)
    print(centroidOfC)

    #compute discriminant lines
    discrimAB = discriminantLine(centroidOfA,centroidOfB)
    discrimAC = discriminantLine(centroidOfA,centroidOfC)
    discrimBC = discriminantLine(centroidOfB,centroidOfC)

    print(discrimAB)
    print(discrimAC)
    print(discrimBC)

    orthogAB = findOrtogonalW(centroidOfA,centroidOfB)
    orthogAC = findOrtogonalW(centroidOfA,centroidOfC)
    orthogBC = findOrtogonalW(centroidOfB,centroidOfC)

    print(orthogAB)

    testdimension = testing_input[0][0]
    sizeOftestA = testing_input[0][1]
    sizeOftestB = testing_input[0][2]
    sizeOftestC = testing_input[0][3]


    classOftestA = testing_input[1:1+sizeOfA]
    classOftestB = testing_input[2 + sizeOfA: 2 + sizeOfA + sizeOfB]
    classOftestC = testing_input[1 + sizeOfA + sizeOfB: ]

    print(len(classOfA))
    print(len(classOfB))
    print(len(classOfC))



    #compute centroids of A,B,C
    centroidOftestA = centroid(classOftestA)
    centroidOftestB = centroid(classOftestB)
    centroidOftestC = centroid(classOftestC)

    TruePosA, TruePosB, TruePosC = 0.0, 0.0, 0.0
    FalsePosA, FalsePosB, FalsePosC = 0.0, 0.0, 0.0
    FalseNegA, FalseNegB, FalseNegC = 0.0, 0.0, 0.0
    TrueNegA, TrueNegB, TrueNegC = 0.0, 0.0, 0.0
    posA, posB, posC = 0.0, 0.0, 0.0
    negA, negB, negC = 0.0, 0.0, 0.0


    truePosRateA, truePosRateB, truePosRateC = 0.0, 0.0, 0.0
    falsePosRateA, falsePosRateB, falsePosRateC = 0.0, 0.0, 0.0
    index = 0
    for i in range(1,76):
        score = classifyPoint(testing_input[i], orthogAB, orthogAC, orthogBC,discrimAB,discrimAC,discrimBC)
        # predicted class A
        if score == 1:
            posA += 1
            negB += 1
            negC += 1
            # is actually A
            if index < sizeOftestA:
                TruePosA += 1
                TrueNegB += 1
                TrueNegC += 1
            elif index >= sizeOftestA and index < sizeOftestA + sizeOftestB:
                FalsePosA += 1
                FalseNegB += 1
                TrueNegC += 1
            else:
                FalsePosA += 1
                FalseNegC += 1
                TrueNegB += 1

        # predicted B
        elif score == 2:
            negA += 1
            posB += 1
            negC += 1
            if index < sizeOftestA:
                FalsePosB += 1
                FalseNegA += 1
                TrueNegC += 1
            elif index >= sizeOftestA and index < sizeOftestA + sizeOftestB:
                TruePosB += 1
                TrueNegA += 1
                TrueNegC += 1
            else:
                FalsePosB += 1
                FalseNegC += 1
                TrueNegA += 1
        # predicted C
        elif score == 3:
            negA += 1
            negB += 1
            posC += 1
            if index < sizeOftestA:
                FalsePosC += 1
                FalseNegA += 1
                TrueNegB += 1
            elif index >= sizeOftestA and index < sizeOftestA + sizeOftestB:
                FalsePosC += 1
                FalseNegB += 1
                TrueNegA += 1
            else:
                TruePosC += 1
                TrueNegA += 1
                TrueNegB += 1
        index += 1

    # class A
    truePosRateA = TruePosA / float(sizeOftestA)
    falsePosRateA = FalsePosA / (float(sizeOftestB) + float(sizeOftestC))
    errorA = (FalsePosA + FalseNegA) / (posA + negA)
    accuracyA = (TruePosA + TrueNegA) / (posA + negA)
    precisionA = TruePosA / posA

    # class B
    truePosRateB = TruePosB / float(sizeOftestB)
    falsePosRateB = FalsePosB / (float(sizeOftestA) + float(sizeOftestC))
    errorB = (FalsePosB + FalseNegB) / (posB + negB)
    accuracyB = (TruePosB + TrueNegB) / (posB + negB)
    precisionB = TruePosB / posB

    # class c
    truePosRateC = TruePosC / float(sizeOftestC)
    falsePosRateC = FalsePosC / (float(sizeOftestA) + float(sizeOftestB))
    errorC = (FalsePosC + FalseNegC) / (posC + negC)
    accuracyC = (TruePosC + TrueNegC) / (posC + negC)
    precisionC = TruePosC / posC



    #results
    print ('True positive rate =', (truePosRateA + truePosRateB + truePosRateC) / 3.0)
    print ('False positive rate =', (falsePosRateA + falsePosRateB + falsePosRateC) / 3.0)
    print ("Error rate = ", (errorA + errorB + errorC) / 3.0)
    print ("Accuracy = ", (accuracyA + accuracyB + accuracyC) / 3.0)
    print ("Precision = ", (precisionA + precisionB + precisionC) / 3.0)

    pass


    