import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
UNITS_INPUT_LAYER = 1024
UNITS_HIDDEN_LAYER = 512


class ModelConfiguration:
    def __init__(self, input_shape, num_classes, lambda_a, lambda_b, dropout_value_1, dropout_value_2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.dropout_value_1 = dropout_value_1
        self.dropout_value_2 = dropout_value_2


    def configuration_base (self):
        """
        Model is built without regularization
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))  # image size 32 × 32 is transferred to a vector with 1024 elements
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu'))  # input layer with 1024 units and ReLU = function activation
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu'))  # hidden layer (with 512 units )
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu'))  # hidden layer (with 512 units )
        model.add(Dense(self.num_classes, activation="softmax"))  # output layer with 27 units and softmax = function activation
        return model

    def configuration_L1_lambda_a (self):
        """
        Model is built with L1 regularization added to all layers except the output layer
        Lambda value is 0.001
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))  # image size 32 × 32 is transferred to a vector with 1024 elements
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(self.lambda_a)))  # input layer with 1024 units and ReLU = function activation
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(self.lambda_a)))  # hidden layer (with 512 units )
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(self.lambda_a)))  # hidden layer (with 512 units )
        model.add(Dense(self.num_classes, activation="softmax"))  # output layer with 27 units and softmax = function activation
        return model

    def configuration_L1_lambda_b (self):
        """
        Model is built with L1 regularization added to all layers except the output layer
        Lambda value is 0.01
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(self.lambda_b)))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(self.lambda_b)))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(self.lambda_b)))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model

    def configuration_L2_lambda_a (self):
        """
        Model is built with L2 regularization added to all layers except the output layer
        Lambda value is 0.001
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_a)))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(self.lambda_a)))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(self.lambda_a)))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model

    def configuration_L2_lambda_b (self):
        """
        Model is built with L2 regularization added to all layers except the output layer
        Lambda value is 0.01
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_b)))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_b)))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_b)))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model

    def configuration_dropout_value_1 (self):
        """
        Model is built with dropout added to all layers except output layer
        Dropout value is p=0.05
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu'))
        model.add(Dropout(self.dropout_value_1))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu'))
        model.add(Dropout(self.dropout_value_1))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu'))
        model.add(Dropout(self.dropout_value_1))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model


    def configuration_L2_lambda_a_dropout_value_2 (self):
        """
        Model is built with both L2 and dropout added to all layers except output layer
        Lambda value is 0.001
        Dropout value is p=0.5
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_a)))
        model.add(Dropout(self.dropout_value_2))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_a)))
        model.add(Dropout(self.dropout_value_2))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_a)))
        model.add(Dropout(self.dropout_value_2))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model

    def configuration_L2_lambda_b_dropout_value_2 (self):
        """
        Model is built with both L2 and dropout added to all layers except output layer
        Lambda value is 0.01
        Dropout value is p=0.5
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(UNITS_INPUT_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_b)))
        model.add(Dropout(self.dropout_value_2))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_b)))
        model.add(Dropout(self.dropout_value_2))
        model.add(Dense(UNITS_HIDDEN_LAYER, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.lambda_b)))
        model.add(Dropout(self.dropout_value_2))
        model.add(Dense(self.num_classes, activation="softmax"))
        return model