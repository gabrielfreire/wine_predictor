import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

# Get data from csv file
def get_data():
    df = pd.read_csv('winequality-red.csv', sep=";")
    return df

# 1 = [1, 0]
# 0 = [0, 1]
def dense_to_one_hot(labels_dense, num_classes=2):
    # Convert class labels from scalars to one-hot vectors
    num_labels = len(labels_dense)
    print(num_labels)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense] = 1
    return labels_one_hot

# METHOD TO ENGINEER THE DATA
def engineer_data():
    
    # GET THE DATA FROM DATASET
    data_frame = get_data()

    df_copy = data_frame.copy()
    # Get rid of the outliers
    df_copy = df_copy[data_frame['total sulfur dioxide'] < 200]
    # make better categories
    bins = [3, 5, 8]
    df_copy['category'] = pd.cut(df_copy['quality'], bins, labels=['Bad', 'Good'])
    df_copy_newcats = df_copy[df_copy['category'].isin(['Bad','Good'])].copy()
    df_copy_newcats['category'] = pd.cut(df_copy_newcats['quality'], bins, labels=[0.0, 1.0])

    all_cols = ['volatile acidity', 'citric acid', 'chlorides', 'residual sugar', 'free sulfur dioxide', 'sulphates', 'total sulfur dioxide', 'density',
       'pH', 'alcohol']
    # feature columns chosen by data analisys on jupyter notebook
    feature_cols = ['volatile acidity','citric acid','total sulfur dioxide','sulphates','alcohol', 'density']
    features = df_copy_newcats[feature_cols].values
    # label (quality)
    labels = df_copy_newcats[['category']].values
    # labels = dense_to_one_hot(labels, num_classes=2)
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    labels = scaler.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test

# ENGINEER THE DATA
X_train, X_test, y_train, y_test = engineer_data()


learning_rate = 0.0001
steps = 100000
    
# input and output placeholders
with tf.variable_scope('input'):
    X = tf.placeholder(dtype=tf.float32)
with tf.variable_scope('output'):
    y = tf.placeholder(dtype=tf.float32)

# BUILD THE MODEL
def build_model():
    # hyperparameters
    n_input = 6
    n_hidden = 80
    n_hidden2 = 80
    n_output = 1

    
    # Weight and activation Operations
    with tf.variable_scope('input_layer'):
        W = tf.get_variable(name="weights1", shape=[n_input, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases1", shape=[n_hidden], initializer=tf.zeros_initializer())
        input_layer = tf.add(tf.matmul(X, W), b)
        input_layer = tf.nn.relu(input_layer)
    
    with tf.variable_scope('hidden_layer1'):
        W2 = tf.get_variable(name="weights2", shape=[n_hidden, n_hidden2], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="biases2", shape=[n_hidden2], initializer=tf.zeros_initializer())
        hidden_layer = tf.add(tf.matmul(input_layer, W2), b2)
        hidden_layer = tf.nn.relu(hidden_layer)
    
    #Softmax is great for categorical data
    with tf.variable_scope('output'):
        W3 = tf.get_variable(name="weights3", shape=[n_hidden2, n_output], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(name="biases3", shape=[n_output], initializer=tf.zeros_initializer())
        output_layer = tf.add(tf.matmul(hidden_layer, W3), b3)
        output_layer = tf.nn.sigmoid(output_layer)
        # output_layer = tf.nn.softmax(output_layer)


    saver = tf.train.Saver()

    return saver, output_layer

# TRAIN THE MODEL
def train_model():
    display_rate = 1000
    saver, pred = build_model()
    # Add an optimizer and create a training operation
        
    with tf.variable_scope('cost'):
        cost = tf.reduce_mean(tf.squared_difference(pred, y))
        # cost = tf.nn.l2_loss(output_layer - y)

    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train = optimizer.minimize(cost)

    # calculate the error
    correct = tf.equal(tf.floor(pred+0.5), y)

    # use the correct to calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            feed_dict = {
                X:X_train,
                y:y_train
            }
            sess.run(train, feed_dict=feed_dict)
            if i % display_rate == 0 :
                loss = sess.run(cost, feed_dict = feed_dict)
                acc = accuracy.eval(feed_dict={X: X_test, y: y_test}) * 100
                print('Training step: {} loss: {} accuracy for test data: {}%'.format(i, loss, acc))

            #save the last step
            if i == steps-1:
                save_path = saver.save(sess, 'model/wine_model.ckpt', global_step = i)
                print("Model saved to: ", save_path, 'on step', i)
        
        prediction = sess.run(pred, feed_dict = { X: X_test, y: y_test })
        acc = accuracy.eval(feed_dict={X: X_test, y: y_test}) * 100
        print('Prediction: ', prediction.tolist())
        print('Accuracy for testing set: {}%'.format(acc))
        print('Accuracy for training set: {}%'.format(accuracy.eval(feed_dict=feed_dict) * 100))

        return prediction, acc

def test_model():
    saver, pred = build_model()
    # calculate the error
    correct = tf.equal(tf.floor(pred+0.5), y)

    # use the correct to calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    with tf.Session() as sess:
        saver.restore(sess, 'model/wine_model.ckpt-99999')
        print(tf.trainable_variables())
        # pred = [v for v in tf.trainable_variables() if v.name == 'output_layer/layer3:0'][0]
        prediction = sess.run(pred, feed_dict={X: X_test})
        acc = accuracy.eval(feed_dict={X: X_test, y: y_test}) * 100
        print('Prediction: ', prediction.tolist())
        plt.hist(y_test,color=['blue'], histtype='step', label='Real')
        plt.hist(prediction,color=['red'], histtype='step', label='Prediction')

        # plt.scatter(prediction, y_test, c=['red', 'blue'])
        plt.legend(prop={'size': 10},loc="upper center")
        plt.title("USING TEST DATA. ACCURACY: {}%".format(acc))
        plt.show()

# prediction, acc = train_model()
test_model()
# plt.hist(y_test,color=['blue'], histtype='step', label='Real')
# plt.hist(prediction,color=['red'], histtype='step', label='Prediction')

# # plt.scatter(prediction, y_test, c=['red', 'blue'])
# plt.legend(prop={'size': 10},loc="upper center")
# plt.title("USING TEST DATA. ACCURACY: {}".format(acc))
# plt.show()