## Oran Ofzer, ID 205633423
## Yevgeni Forost, ID 314391558
import AdaBoost
import matplotlib.pyplot as plot
import pandas as pd

if __name__ == '__main__':
    filename = 'testData.csv'
    data = pd.read_csv(filename, header=None)

    dataset = data.rename(columns={0: 'label'})
# - Increasing the number of stumps will help increase the accuracy, but will take longer threshold serves as a stop
    # condition if we've reached that accuracy in % (can modify depends on dataset)
# - Adaboost will stop when reached num_of_base_learners or the threshold, which ever comes first
# - train_data_size and test_data_size can be modified to test smaller size of data (0...1, 1 = all of the data)
# - same_points decides weather the points used in test data will be of those used in train data
    number_of_base_learners = 150
    threshold = 100
    train_data_size = 1  # please choose between 0 and 1 (0.8, 0.9 and so)
    test_data_size = 0.2 # please choose between 0 and 1 (0.8, 0.9 and so)
    same_points = True
    # Changes same_points in case that the train data will contain all of the data (in that case we cant choose
    # different points in test data)
    if train_data_size == 1:
        same_points = True  # true for different points in the test data and the train data

# Train data and test data will be modified, based on train_data_size, test_data_size and same_points that we chose
    data_train = dataset[:int(train_data_size * len(dataset))]
    if same_points:
        data_test = dataset[:int(test_data_size * len(dataset))]
    else:
        data_test = dataset[int((1 - test_data_size) * len(dataset)):]
    
    fig = plot.figure(figsize=(15, 15))
    plot1 = fig.add_subplot(111)

# Train data will be sent as the first argument, test data as the third.
# data_test can be used as the third argument to test new points that were not used in the trained data
    model = AdaBoost.AdaBoost(data_train, number_of_base_learners, data_test, threshold, same_points)
    model.fit()
    model.predict()

    plot1.plot(range(len(model.accuracy)), model.accuracy, '-b')
    plot1.set_xlabel('Number of stumps')
    plot1.set_ylabel('Accuracy %')

    print("The prediction list is:")
    print(model.predictions)
    print('With ', model.train_num_of_stumps_used, ' decision stumps used for the training, we receive a training '
                                                   'accuracy of ', model.train_accuracy * 100, '%')
    print('With ', model.num_of_stumps_used, ' decision stumps used for the prediction, we receive a prediction '
                                             'accuracy of ', model.accuracy[-1] * 100, '%')

    plot.show()
