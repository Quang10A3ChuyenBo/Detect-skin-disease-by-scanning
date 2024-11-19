import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, num_classes, l1_reg=0.001, num_blocks=2):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.conv_blocks = []
        for _ in range(num_blocks):
            block = keras.Sequential(
                [
                    layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l1(l1_reg)),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPool2D(2, 2),
                ]
            )
            self.conv_blocks.append(block)


        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l1(l1_reg))
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l1(l1_reg))
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=False):

        x = inputs
        for block in self.conv_blocks:
            x = block(x, training=training)


        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)