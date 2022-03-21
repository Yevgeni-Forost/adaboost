## Oran Ofzer, ID 205633423
## Yevgeni Forost, ID 314391558
from sklearn.tree import DecisionTreeClassifier
import numpy as nmp
import pandas as pd

class AdaBoost:
    def __init__(self, data_set, T, test_data, target_acc, same):  # Initialize all needed data for Adaboost
        self.data_set = data_set
        self.T = T
        self.test_data = test_data
        self.stumps = None  # Decision Stumps
        self.alphas = None  # Decision stumps amount of say
        self.accuracy = []  # Accuracy of the test data predictions measured for each iteration
        self.train_accuracy = 0  # Training accuracy of the model
        self.target_acc = target_acc/100  # The max accuracy of the hypothesis on the test data
        self.predictions = None  # The model predictions table of the test data
        self.num_of_stumps_used = 0  # Number of classifiers used for the prediction of the model
        self.train_num_of_stumps_used = 0  # Number of classifiers used for the training of the model
        self.same = same  # a boolean that shows if the test data is different from train data

    def fit(self):
        stumps = []
        alphas = []
        
        X = self.data_set.drop(columns=['label'], axis=1)  # Create Dataset without labels col
        Y = self.data_set['label'].where(self.data_set['label'] == 1, -1)  # change 0 labels to -1 for sign
        
        Process = pd.DataFrame(Y.copy())
        Process['weights'] = 1/len(self.data_set)  # Initialize data weights to 1/n

        predictions = []
        
        for t in range(self.T):
            min_error_rate = 1
            
            for i in range(len(self.data_set.columns)-2):
                stump_gen = DecisionTreeClassifier(max_depth=1)  # Create a decision stump
                stump = stump_gen.fit(X, Y, sample_weight=nmp.array(Process['weights']))  # send dataset and labels to
                # generate to create a model
                stump_predictions = stump.predict(X)
                Process['curr_predictions'] = stump_predictions
                Process['curr_misclassified'] = nmp.where(Process['curr_predictions'] != Process['label'], 1, 0)
                error_rate = sum(Process['weights']*Process['curr_misclassified'])
                
                if min_error_rate > error_rate:
                    if len(stumps) > t:
                        stumps.pop()
                    min_error_rate = error_rate
                    stumps.append(stump)
                    # Check correctly classified / misclassified points for stump t
                    # Save them in Process arr
                    Process['predictions'] = stump_predictions
                    Process['success'] = nmp.where(Process['predictions'] == Process['label'], 1, 0)
                    Process['misclassified'] = nmp.where(Process['predictions'] != Process['label'], 1, 0)
                    
            # Added 0.0000000001 to the min_error_rate in case of division by zero       
            alpha = 0.5 * (nmp.log((1-min_error_rate) / (min_error_rate + 0.0000000001))) 
            alphas.append(alpha)
            
            Process['weights'] = nmp.where(Process['predictions'] != Process['label'], 0.5 * Process['weights']
                                           / min_error_rate, 0.5 * Process['weights'] / (1 - min_error_rate))
                
            self.alphas = alphas
            self.stumps = stumps

            predictions.append(alpha * stump_predictions)
            # Calculate accuracy by iterating over all stumps and checking where each stump is equal to Y label and
            # dividing by dataset size to measure accuracy
            temp_pred = nmp.sign(nmp.sum(nmp.array(predictions), axis=0))
            curr_accuracy = nmp.sum(temp_pred == Y.values) / len(predictions[0])

            # Calculate the number of stumps used to train the model
            self.train_num_of_stumps_used += 1

            # stop the training if reached the max accuracy (used in cases where the training data and the test data
            # are the same, in those cases we do not need to train more)
            if self.same and curr_accuracy == 1:
                break

        # saves the final training accuracy of the model
        self.train_accuracy = curr_accuracy
    
    # predict function to test the model on the training data
    def predict(self):
        X_test = self.test_data.drop(columns=['label'], axis=1)
        Y_test = self.test_data['label'].where(self.test_data['label'] == 1, -1)
        
        predictions = []
        
        for alpha, stump in zip(self.alphas, self.stumps):
            prediction = alpha * stump.predict(X_test)
            predictions.append(prediction)
            # Calculate accuracy by iterating over all stumps and checking where each stump's prediction is equal to Y
            # label and dividing by dataset size to measure accuracy
            temp_pred = nmp.sign(nmp.sum(nmp.array(predictions), axis=0))  # The final H(x) of the model (on the last
            # iteration)
            curr_accuracy = nmp.sum(temp_pred == Y_test.values) / len(predictions[0])
            self.accuracy.append(curr_accuracy)
            # Calculates the number of classifiers used to predict on the test data
            self.num_of_stumps_used += 1
            # Stop the calculations if the desired accuracy reached
            if curr_accuracy >= self.target_acc:
                print("Reached required accuracy")
                break

        # Saves the prediction list in the form of a table
        pred = pd.DataFrame(Y_test.copy())
        pred['label'] = temp_pred
        self.predictions = pred
                                     