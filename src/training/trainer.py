"""
Autoencoder Trainer
===================

Handles training loop with early stopping and model checkpointing.
Saves training history and best models.

Features:
    - Early stopping (patience=10 epochs)
    - Model checkpointing (saves best validation loss)
    - Training history logging (loss curves)
    - Progress reporting

Reference: CANShield training pipeline
"""

import json
import numpy as np
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_autoencoder(
    autoencoder,
    X_train,
    output_dir,
    time_scale,
    epochs=500,
    batch_size=128,
    validation_split=0.1,
    verbose=1
):
    """
    Train autoencoder with early stopping and checkpointing.
    
    Args:
        autoencoder (keras.Model): Compiled autoencoder
        X_train (np.ndarray): Training windows (N, 50, 15, 1)
        output_dir (Path|str): Directory to save models and history
        time_scale (int): Time scale identifier (1, 5, 10, 20, 50)
        epochs (int): Maximum training epochs (default: 500)
        batch_size (int): Training batch size (default: 128)
        validation_split (float): Fraction for validation (default: 0.1)
        verbose (int): Verbosity level (0=silent, 1=progress, 2=epoch)
    
    Returns:
        tuple: (trained_model, history_dict)
            - trained_model: Best model (restored from checkpoint)
            - history_dict: Training history (loss, val_loss, etc.)
    
    Workflow:
        1. Setup callbacks (EarlyStopping + ModelCheckpoint)
        2. Train model (input = output for autoencoder!)
        3. Save final model + training history
        4. Return trained model
    
    Example:
        >>> model = build_2d_cnn_autoencoder(50, 15)
        >>> model = compile_autoencoder(model)
        >>> trained, history = train_autoencoder(
        ...     model, X_train, 'results/training/models', time_scale=1
        ... )
        >>> print(f"Final loss: {history['loss'][-1]:.6f}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # SETUP CALLBACKS
    # ========================================================================
    
    # Checkpoint: Save best model (lowest validation loss)
    checkpoint_path = output_dir / f'autoencoder_T{time_scale}_best.h5'
    
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1 if verbose > 0 else 0,
        save_weights_only=False
    )
    
    # Early stopping: Stop if no improvement for 10 epochs
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1 if verbose > 0 else 0
    )
    
    callbacks = [checkpoint, early_stop]
    
    # ========================================================================
    # PRINT TRAINING INFO
    # ========================================================================
    
    if verbose > 0:
        print("\n" + "="*80)
        print(f"TRAINING AUTOENCODER - TIME SCALE T={time_scale}")
        print("="*80)
        print(f"\nDataset:")
        print(f"  Total samples:      {X_train.shape[0]:,}")
        print(f"  Input shape:        {X_train.shape[1:]}")
        print(f"  Data range:         [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"  Data mean:          {X_train.mean():.4f}")
        
        val_samples = int(X_train.shape[0] * validation_split)
        train_samples = X_train.shape[0] - val_samples
        
        print(f"\nSplit:")
        print(f"  Training samples:   {train_samples:,} ({(1-validation_split)*100:.0f}%)")
        print(f"  Validation samples: {val_samples:,} ({validation_split*100:.0f}%)")
        
        print(f"\nHyperparameters:")
        print(f"  Max epochs:         {epochs}")
        print(f"  Batch size:         {batch_size}")
        print(f"  Early stop patience: 10 epochs")
        
        print(f"\nOutput:")
        print(f"  Checkpoint:         {checkpoint_path.name}")
        print(f"  Final model:        autoencoder_T{time_scale}.h5")
        print(f"  History:            history_T{time_scale}.json")
        print()
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    if verbose > 0:
        print("="*80)
        print("Starting training...")
        print("="*80)
    
    # Autoencoder training: Input = Output (reconstruct itself!)
    history = autoencoder.fit(
        X_train, X_train,  # Key: input = target for autoencoder
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True
    )
    
    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================
    
    # Save final model (may differ from best if early stopping didn't trigger)
    final_model_path = output_dir / f'autoencoder_T{time_scale}.h5'
    autoencoder.save(final_model_path)
    
    if verbose > 0:
        print("\n" + "="*80)
        print("Training complete!")
        print("="*80)
        print(f"✓ Final model saved:      {final_model_path}")
        print(f"✓ Best model saved:       {checkpoint_path}")
    
    # Save training history as JSON
    history_path = output_dir.parent / 'histories' / f'history_T{time_scale}.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert history to serializable format
    history_dict = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    
    # Add metadata
    history_dict['metadata'] = {
        'time_scale': time_scale,
        'total_epochs': len(history_dict['loss']),
        'batch_size': batch_size,
        'validation_split': validation_split,
        'total_samples': int(X_train.shape[0]),
        'final_loss': float(history_dict['loss'][-1]),
        'final_val_loss': float(history_dict['val_loss'][-1]),
        'best_val_loss': float(min(history_dict['val_loss'])),
        'best_epoch': int(np.argmin(history_dict['val_loss']) + 1)
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    if verbose > 0:
        print(f"✓ Training history saved: {history_path}")
        
        print(f"\nTraining Summary:")
        print(f"  Total epochs:       {history_dict['metadata']['total_epochs']}")
        print(f"  Best epoch:         {history_dict['metadata']['best_epoch']}")
        print(f"  Final loss:         {history_dict['metadata']['final_loss']:.6f}")
        print(f"  Final val_loss:     {history_dict['metadata']['final_val_loss']:.6f}")
        print(f"  Best val_loss:      {history_dict['metadata']['best_val_loss']:.6f}")
        print()
    
    return autoencoder, history_dict


def load_trained_model(model_path):
    """
    Load a trained autoencoder model.
    
    Args:
        model_path (Path|str): Path to .h5 model file
    
    Returns:
        keras.Model: Loaded model
    
    Example:
        >>> model = load_trained_model('results/training/models/autoencoder_T1.h5')
        >>> predictions = model.predict(X_test)
    """
    from tensorflow.keras.models import load_model
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    return model


def evaluate_model(autoencoder, X_test, verbose=1):
    """
    Evaluate autoencoder on test data.
    
    Args:
        autoencoder: Trained model
        X_test (np.ndarray): Test windows (N, 50, 15, 1)
        verbose (int): Verbosity
    
    Returns:
        dict: Evaluation metrics (loss, mae)
    """
    if verbose > 0:
        print("\nEvaluating model on test data...")
    
    results = autoencoder.evaluate(X_test, X_test, verbose=verbose)
    
    metrics = {
        'loss': float(results[0]),
        'mae': float(results[1])
    }
    
    if verbose > 0:
        print(f"\nTest Results:")
        print(f"  Loss (MSE): {metrics['loss']:.6f}")
        print(f"  MAE:        {metrics['mae']:.6f}")
    
    return metrics


# Quick test function
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from training.autoencoder_builder import build_2d_cnn_autoencoder, compile_autoencoder
    
    print("="*80)
    print("Testing Trainer Module")
    print("="*80)
    
    # Create dummy data
    print("\nCreating dummy training data...")
    np.random.seed(42)
    X_dummy = np.random.rand(1000, 50, 15, 1).astype('float32')
    
    # Build and compile model
    print("Building autoencoder...")
    model = build_2d_cnn_autoencoder(50, 15)
    model = compile_autoencoder(model)
    
    # Train (just 5 epochs for testing)
    print("\nTraining (5 epochs test)...")
    trained_model, history = train_autoencoder(
        autoencoder=model,
        X_train=X_dummy,
        output_dir=Path('test_output'),
        time_scale=1,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("\n✓ Trainer test complete!")
    print(f"  Final loss: {history['metadata']['final_loss']:.6f}")
    print(f"  Final val_loss: {history['metadata']['final_val_loss']:.6f}")
