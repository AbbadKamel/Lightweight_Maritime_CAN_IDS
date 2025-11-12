"""
Transfer learning utilities
Based on CANShield Phase 2
"""
from tensorflow import keras
from typing import Optional


class TransferLearner:
    """Transfer learning utilities for autoencoders"""
    
    @staticmethod
    def load_pretrained_weights(model: keras.Model, 
                               pretrained_model_path: str,
                               freeze_layers: bool = False) -> keras.Model:
        """
        Load pretrained weights from previous sampling period model
        
        Args:
            model: Target model to initialize
            pretrained_model_path: Path to pretrained model
            freeze_layers: Whether to freeze loaded layers
        
        Returns:
            Model with transferred weights
        """
        # Load pretrained model
        pretrained_model = keras.models.load_model(pretrained_model_path)
        
        # Transfer weights layer by layer
        for layer, pretrained_layer in zip(model.layers, pretrained_model.layers):
            try:
                layer.set_weights(pretrained_layer.get_weights())
                if freeze_layers:
                    layer.trainable = False
            except ValueError:
                # Skip if shapes don't match
                print(f"Warning: Could not transfer weights for layer {layer.name}")
                continue
        
        return model
    
    @staticmethod
    def fine_tune(model: keras.Model, 
                 unfreeze_last_n_layers: Optional[int] = None) -> keras.Model:
        """
        Fine-tune model by unfreezing last N layers
        
        Args:
            model: Model to fine-tune
            unfreeze_last_n_layers: Number of last layers to unfreeze
        
        Returns:
            Model ready for fine-tuning
        """
        if unfreeze_last_n_layers is None:
            # Unfreeze all layers
            for layer in model.layers:
                layer.trainable = True
        else:
            # Unfreeze last N layers
            for layer in model.layers[:-unfreeze_last_n_layers]:
                layer.trainable = False
            for layer in model.layers[-unfreeze_last_n_layers:]:
                layer.trainable = True
        
        return model


if __name__ == "__main__":
    print("TransferLearner ready")
