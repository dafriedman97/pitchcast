from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner

class CustomHyperModel(keras_tuner.HyperModel):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def build(self, hp):
        # Input layer
        model = keras.Sequential(name='sequential_nn')
        model.add(keras.Input(shape=(self.input_shape,), name='input'))
        
        # First hidden layer
        l1_nodes = hp.Int("l1_nodes", min_value=48, max_value=64, step=16)
        l1_dropout = hp.Float("l1_dropout", min_value=0.0, max_value=0.1, step=0.05)
        model.add(layers.Dense(l1_nodes, name='dense_1'))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(l1_dropout))
        
        # Second hidden layer
        l2_nodes = hp.Int("l2_nodes", min_value=48, max_value=64, step=16)
        l2_dropout = hp.Float("l2_dropout", min_value=0.0, max_value=0.1, step=0.05)
        model.add(layers.Dense(l2_nodes, name='dense_2'))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(l2_dropout))
        
        # Output layer
        model.add(layers.Dense(self.output_shape, name="output"))
        model.add(layers.Activation("softmax"))
        
        # Compile
        lr = hp.Choice("lr", [5e-4, 1e-3])
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        return model