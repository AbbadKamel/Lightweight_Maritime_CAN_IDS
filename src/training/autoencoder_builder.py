"""
Autoencoder Architecture Builder
=================================

Builds 2D-CNN autoencoder following CANShield paper architecture.
Adapted for 15 maritime signals (original used 20 signals).

Architecture:
    Input: (batch, 50, 15, 1) - 50 timesteps × 15 signals × 1 channel
    
    Encoder:
        - ZeroPadding + Conv2D(32, 5×5) + LeakyReLU + MaxPool(2×2)
        - Conv2D(16, 5×5) + LeakyReLU + MaxPool(2×2)
        - Conv2D(16, 3×3) + LeakyReLU + MaxPool(2×2) → Bottleneck
    
    Decoder:
        - Conv2D(16, 3×3) + LeakyReLU + UpSample(2×2)
        - Conv2D(16, 5×5) + LeakyReLU + UpSample(2×2)
        - Conv2D(32, 5×5) + LeakyReLU + UpSample(2×2)
        - Conv2D(1, 3×3, sigmoid) + Cropping
    
    Output: (batch, 50, 15, 1) - reconstructed input

Reference: CANShield paper (2023)
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    LeakyReLU, 
    MaxPooling2D, 
    UpSampling2D,
    ZeroPadding2D, 
    Cropping2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


def build_2d_cnn_autoencoder(time_step=50, num_signals=15):
    """
    Build 2D-CNN autoencoder architecture.
    
    Treats time×signals as 2D image for spatial-temporal feature learning.
    
    Args:
        time_step (int): Window length (default: 50)
        num_signals (int): Number of CAN signals (default: 15)
    
    Returns:
        keras.Sequential: Uncompiled autoencoder model
    
    Example:
        >>> model = build_2d_cnn_autoencoder(50, 15)
        >>> model.summary()
        Total params: ~50,000
    """
    in_shape = (time_step, num_signals, 1)  # (50, 15, 1)
    
    autoencoder = Sequential(name='CANShield_Autoencoder')
    
    # ========================================================================
    # ENCODER (Compress input to bottleneck)
    # ========================================================================
    
    # Zero-padding to handle edge effects
    autoencoder.add(ZeroPadding2D(
        padding=(2, 2),
        input_shape=in_shape,
        name='zero_pad_input'
    ))
    
    # --- Encoder Layer 1 ---
    autoencoder.add(Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        name='enc_conv1'
    ))
    autoencoder.add(LeakyReLU(alpha=0.2, name='enc_leaky1'))
    autoencoder.add(MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        name='enc_pool1'
    ))
    
    # --- Encoder Layer 2 ---
    autoencoder.add(Conv2D(
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        name='enc_conv2'
    ))
    autoencoder.add(LeakyReLU(alpha=0.2, name='enc_leaky2'))
    autoencoder.add(MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        name='enc_pool2'
    ))
    
    # --- Encoder Layer 3 (Bottleneck) ---
    autoencoder.add(Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name='enc_conv3_bottleneck'
    ))
    autoencoder.add(LeakyReLU(alpha=0.2, name='enc_leaky3'))
    autoencoder.add(MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        name='enc_pool3_bottleneck'
    ))
    
    # ========================================================================
    # DECODER (Reconstruct from bottleneck)
    # ========================================================================
    
    # --- Decoder Layer 1 ---
    autoencoder.add(Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name='dec_conv1'
    ))
    autoencoder.add(LeakyReLU(alpha=0.2, name='dec_leaky1'))
    autoencoder.add(UpSampling2D(
        size=(2, 2),
        name='dec_upsample1'
    ))
    
    # --- Decoder Layer 2 ---
    autoencoder.add(Conv2D(
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        name='dec_conv2'
    ))
    autoencoder.add(LeakyReLU(alpha=0.2, name='dec_leaky2'))
    autoencoder.add(UpSampling2D(
        size=(2, 2),
        name='dec_upsample2'
    ))
    
    # --- Decoder Layer 3 ---
    autoencoder.add(Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        name='dec_conv3'
    ))
    autoencoder.add(LeakyReLU(alpha=0.2, name='dec_leaky3'))
    autoencoder.add(UpSampling2D(
        size=(2, 2),
        name='dec_upsample3'
    ))
    
    # --- Output Layer ---
    autoencoder.add(Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same',
        name='dec_output'
    ))
    
    # ========================================================================
    # CROPPING (Remove padding to match original input shape)
    # ========================================================================
    
    # Calculate cropping needed
    temp_shape = autoencoder.output_shape
    
    # Time dimension cropping
    diff_time = temp_shape[1] - in_shape[0]
    top = diff_time // 2
    bottom = diff_time - top
    
    # Signal dimension cropping
    diff_signal = temp_shape[2] - in_shape[1]
    left = diff_signal // 2
    right = diff_signal - left
    
    autoencoder.add(Cropping2D(
        cropping=((top, bottom), (left, right)),
        name='crop_to_original'
    ))
    
    return autoencoder


def compile_autoencoder(autoencoder, learning_rate=0.0002):
    """
    Compile autoencoder with Adam optimizer and MSE loss.
    
    Args:
        autoencoder (keras.Model): Uncompiled model
        learning_rate (float): Adam learning rate (default: 0.0002)
    
    Returns:
        keras.Model: Compiled model ready for training
    
    Configuration (from CANShield paper):
        - Optimizer: Adam(lr=0.0002, beta_1=0.5, beta_2=0.99)
        - Loss: MeanSquaredError (reconstruction quality)
        - Metrics: MAE (Mean Absolute Error for monitoring)
    """
    opt = Adam(
        learning_rate=learning_rate,
        beta_1=0.5,
        beta_2=0.99
    )
    
    autoencoder.compile(
        optimizer=opt,
        loss=MeanSquaredError(),
        metrics=['mae']  # Mean Absolute Error for additional monitoring
    )
    
    return autoencoder


def get_model_info(autoencoder):
    """
    Get model architecture information.
    
    Args:
        autoencoder: Compiled or uncompiled model
    
    Returns:
        dict: Model statistics (params, layers, input/output shapes)
    """
    total_params = autoencoder.count_params()
    num_layers = len(autoencoder.layers)
    input_shape = autoencoder.input_shape
    output_shape = autoencoder.output_shape
    
    return {
        'total_params': total_params,
        'num_layers': num_layers,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'trainable': autoencoder.trainable
    }


# Quick test function
if __name__ == '__main__':
    print("="*80)
    print("Testing Autoencoder Builder")
    print("="*80)
    
    # Build model
    print("\nBuilding 2D-CNN autoencoder...")
    model = build_2d_cnn_autoencoder(time_step=50, num_signals=15)
    
    # Compile
    print("Compiling model...")
    model = compile_autoencoder(model)
    
    # Show summary
    print("\nModel Summary:")
    print("="*80)
    model.summary()
    
    # Get info
    info = get_model_info(model)
    print("\n" + "="*80)
    print("Model Information:")
    print("="*80)
    for key, val in info.items():
        print(f"  {key}: {val}")
    
    print("\n✓ Autoencoder builder test complete!")
