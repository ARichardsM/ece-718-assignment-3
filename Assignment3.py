import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# PULL DATA FROM LOAD_BREAST_CANCER
def loadBreastCancer():
    breaCan = load_breast_cancer()

    # Pull data from the dataset
    breastCancerFeats = np.array(breaCan.feature_names)
    breastCancerData = np.array(breaCan.data)
    breastCancerTarget = 1 - np.array(breaCan.target) # ['benign' 'malignant']

    # Split loaded data into train and test sets
    trainData = breastCancerData[:250]
    trainTarget = breastCancerTarget[:250]
    testData = breastCancerData[250:]
    testTarget = breastCancerTarget[250:]
    return breastCancerFeats, trainData, testData, trainTarget, testTarget

# FEATURE STANDARDIZATION
def featStand(X_train, X_valid):
    sc = StandardScaler()

    X_train_Regression = sc.fit_transform(X_train)
    X_valid_Regression = sc.transform(X_valid)
    return X_train_Regression, X_valid_Regression

# ANALYTICAL SOLUTION FUNCTION
def anSol(X_value, t_value):
    w = np.matmul(np.transpose(X_value), X_value)
    w = np.matmul(np.linalg.pinv(w), np.transpose(X_value)) # Psuedo Inverse is used due to singular matrices
    w = np.matmul(w, t_value)

    return w

# LINEAR CLASSIFIER FUNCTION
def linClassifer(z_array, threshold):
    # Prepare an array for the classified features
    classFeat = np.zeros([z_array.size])

    # Check array elements against the threshold
    for i in range(z_array.size):
        if (z_array[i] >= threshold):
            classFeat[i] = 1

    return classFeat

# RECEIVER OPERATING CHARACTERISTICS FUNCTION
def ROCFunction(pred, target):
    # Prepare variables to hold categories
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Check array elements against the threshold
    for i in range(pred.size):
        if (pred[i] == 1) and (target[i] == 1):
            TP = TP + 1
        elif (pred[i] == 1) and (target[i] == 0):
            FP = FP + 1
        elif (pred[i] == 0) and (target[i] == 0):
            TN = TN + 1
        else:
            FN = FN + 1

    # Calculate the ROC
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity

# CALCULATE EUCLIDEAN DISTANCES
def distMeas(neighbours, point):
    distance = np.zeros([neighbours.shape[0]])
    for i in range(neighbours.shape[0]):
        for j in range(point.size):
            distance[i] = distance[i] + pow((neighbours[i][j] - point[j]),2)
    return distance

# DETERMINE NEAREST NEIGHBOUR PREDICTION
def kNearClassPred(data, target, point, k):
    predicition = -1;

    # Get distance of data array from point
    distances = distMeas(data, point)
    distTargets = np.sort(distances)[:k]
    distClass = np.zeros(k)

    # Find the target value associated with the nearest points
    for i in range(distances.size):
        for j in range(k):
            if (distances[i] == distTargets[j]):
                distClass[j] = target[i]

    # Determine predicted class from nearest neighbours
    if ((sum(distClass)/k) > 0.5):
        predicition = 1;
    else:
        predicition = 0;

    return predicition

# LINEAR CLASSIFIER
def mainLinClass(trainData, testData, trainTarget, testTarget):
    w = anSol(trainData, trainTarget)
    z = np.matmul(np.transpose(w), np.transpose(testData))
    sortedZ = np.sort(z)
    sensArr = np.zeros([z.size])
    specArr = np.zeros([z.size])
    count = 0

    # Set inital minimum misclassification threshold to the first value in sorted z
    miniThres = sortedZ[0]
    predClass = linClassifer(z, miniThres)
    miniThresErr = sum(abs(testTarget - predClass))
    miniThresI = 0

    # Find minimum misclassification threshold and ROC values
    for i in sortedZ:
        predClass = linClassifer(z, i)
        sensArr[count], specArr[count] = ROCFunction(predClass, testTarget)
        if(sum(abs(testTarget - predClass)) < miniThresErr):
            miniThres = i
            miniThresI = count
            miniThresErr = sum(abs(testTarget - predClass))
        count = count + 1
    specArr = 1 - specArr

    # Plot ROC curve
    mpl.plot(specArr, sensArr, 'og')
    mpl.plot(specArr[miniThresI], sensArr[miniThresI], 'or')
    mpl.show()

    return miniThresErr

# SEPERATE DATA BASED ON FOLDS
def crossFold(trainData, trainTarget, K, j):
    start = int((trainTarget.size/K)*j)     # Array Index for start of training fold
    end = int((trainTarget.size/K)*(j+1))   # Array Index for end of training fold

    # Seperate the data used for validation from the data used for training
    trainFoldTarget = np.concatenate([trainTarget[:start], trainTarget[end:]])
    trainFoldData = np.concatenate([trainData[:start], trainData[end:]])

    valData = trainData[start:end]
    valTarget = trainTarget[start:end]

    return trainFoldData, trainFoldTarget, valData, valTarget

# K-NEAREST NEIGHBOUR
def kNearNeigh(data, target, pArray, pTarget, folds):
    # Determine validation errors
    error = np.zeros((5, 1))
    count = 0
    for k in [1, 3, 5, 7, 9]:
        for j in range(folds):
            foldTrainData, foldTrainTarget, valData, valTarget = crossFold(data, target, folds, j)

            # Determine prediciton for each point in the validation array
            pred = np.empty(valTarget.size)
            for i in range(valTarget.size):
                pred[i] = kNearClassPred(foldTrainData, foldTrainTarget, valData[i], k)

            error[count] = error[count] + sum(abs(valTarget - pred))
        error[count] = error[count]/folds
        count = count + 1

    # Calculate test predicition using k = 3
    testPred = np.empty(pTarget.size)
    for i in range(pTarget.size):
        testPred[i] = kNearClassPred(data, target, pArray[i], 3)

    # Return the error
    return sum(abs(pTarget - testPred))

# SCIKIT_LEARN K-NEAREST NEIGHBOUR
def kNearNeighScikitLearn(data, target, pArray, pTarget, folds):
    # Determine validation errors
    error = np.zeros((5, 1))
    count = 0
    for k in [1, 3, 5, 7, 9]:
        for j in range(folds):
            foldTrainData, foldTrainTarget, valData, valTarget = crossFold(data, target, folds, j)

            # Determine prediciton for each point in the validation array
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(foldTrainData, foldTrainTarget)

            pred = np.empty(valTarget.size)
            for i in range(valTarget.size):
                pred[i] = neigh.predict([valData[i]])

            error[count] = error[count] + sum(abs(valTarget - pred))
        error[count] = error[count]/folds
        count = count + 1

    # Calculate test predicition using k = 3
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(data, target)
    testPred = np.empty(pTarget.size)
    for i in range(pTarget.size):
        testPred[i] = neigh.predict([pArray[i]])

    # Return the error
    return sum(abs(pTarget - testPred))

# SCIKIT_LEARN LINEAR CLASSIFIER
def mainLinClassScikitLearn(trainData, testData, trainTarget, testTarget):
    logReg = LogisticRegression(multi_class='multinomial')
    logReg.fit(trainData, trainTarget)

    z = np.empty(testTarget.size)
    for i in range(testTarget.size):
        z[i] = logReg.predict_proba([testData[i]])[0][1]

    sortedZ = np.sort(z)
    sensArr = np.zeros([z.size])
    specArr = np.zeros([z.size])
    count = 0

    # Set inital minimum misclassification threshold to the first value in sorted z
    miniThres = sortedZ[0]
    predClass = linClassifer(z, miniThres)
    miniThresErr = sum(abs(testTarget - predClass))
    miniThresI = 0

    # Find minimum misclassification threshold and ROC values
    for i in sortedZ:
        predClass = linClassifer(z, i)
        sensArr[count], specArr[count] = ROCFunction(predClass, testTarget)
        if(sum(abs(testTarget - predClass)) < miniThresErr):
            miniThres = i
            miniThresI = count
            miniThresErr = sum(abs(testTarget - predClass))
        count = count + 1
    specArr = 1 - specArr

    # Plot ROC curve
    mpl.plot(specArr, sensArr, 'og')
    mpl.plot(specArr[miniThresI], sensArr[miniThresI], 'or')
    mpl.show()

    return miniThresErr

breastCancerFeats, trainData, testData, trainTarget, testTarget = loadBreastCancer()
trainData, testData = featStand(trainData, testData)

err1 = mainLinClass(trainData, testData, trainTarget, testTarget)
print(err1)
err2 = kNearNeigh(trainData, trainTarget, testData, testTarget, 10)
print(err2)
err3 = kNearNeighScikitLearn(trainData, trainTarget, testData, testTarget, 10)
print(err3)
err4 = mainLinClassScikitLearn(trainData, testData, trainTarget, testTarget)
print(err4)