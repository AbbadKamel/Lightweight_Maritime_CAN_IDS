"""
CNN Autoencoder architecture
Based on CANShield Phase 2
5 convolutional layers: 32→16→16→32→1
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple


class CNNAutoencoder:
    """CNN Autoencoder for signal reconstruction"""
    
    def __init__(self, input_shape: Tuple[int, int]):
        """
        Args:
            input_shape: (height, width) of input views
        """
        self.input_shape = input_shape
        self.model = None
    
    def build(self) -> keras.Model:
        """
        Build CNN autoencoder architecture
        5 conv layers: 32→16→16→32→1
        
        Returns:
            Keras model
        """
        inputs = keras.Input(shape=(*self.input_shape, 1))
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Bottleneck
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Decoder
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        outputs = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_Autoencoder')
        
        return self.model
    
    def compile(self, learning_rate: float = 0.0002):
        """
        Compile model with optimizer and loss
        
        Args:
            learning_rate: Adam optimizer learning rate (default 0.0002)
        """
        if self.model is None:
            self.build()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build()
        self.model.summary()
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)


if __name__ == "__main__":
    # Test
    autoencoder = CNNAutoencoder(input_shape=(200, 50))
    autoencoder.build()
    autoencoder.summary()
    print("\nCNNAutoencoder ready")
