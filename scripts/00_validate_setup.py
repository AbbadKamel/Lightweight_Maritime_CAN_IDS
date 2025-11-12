#!/usr/bin/env python3
"""
Validate Project Setup
Tests all modules BEFORE collecting real data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test if all imports work"""
    print("="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        # Core dependencies
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        
        import yaml
        print("✓ YAML")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        
        import seaborn
        print(f"✓ Seaborn {seaborn.__version__}")
        
        print("\n✅ All dependencies installed correctly!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Run: pip install -r requirements.txt\n")
        return False


def test_project_modules():
    """Test if project modules load"""
    print("="*60)
    print("TEST 2: Project Modules")
    print("="*60)
    
    try:
        # Utils
        from utils.config_loader import ConfigLoader
        print("✓ ConfigLoader")
        
        from utils.io_utils import load_csv, save_csv
        print("✓ IO Utils")
        
        from utils.logger import setup_logger
        print("✓ Logger")
        
        # Initialization
        from initialization.signal_selector import SignalSelector
        print("✓ SignalSelector")
        
        from initialization.correlation_analyzer import CorrelationAnalyzer
        print("✓ CorrelationAnalyzer")
        
        from initialization.signal_reorderer import SignalReorderer
        print("✓ SignalReorderer")
        
        # Preprocessing
        from preprocessing.n2k_decoder import N2KDecoder
        print("✓ N2KDecoder")
        
        from preprocessing.fifo_queue import FIFOQueue
        print("✓ FIFOQueue")
        
        from preprocessing.view_builder import ViewBuilder
        print("✓ ViewBuilder")
        
        from preprocessing.normalizer import Normalizer
        print("✓ Normalizer")
        
        # Models
        from models.autoencoder import CNNAutoencoder
        print("✓ CNNAutoencoder")
        
        from models.transfer_learning import TransferLearner
        print("✓ TransferLearner")
        
        print("\n✅ All project modules load successfully!\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ Module import error: {e}")
        print("Check if you're in the project root directory\n")
        return False


def test_configuration():
    """Test if configuration files are valid"""
    print("="*60)
    print("TEST 3: Configuration Files")
    print("="*60)
    
    try:
        from utils.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        # Test each config
        n2k_config = loader.load("n2k_config")
        print(f"✓ n2k_config.yaml - {len(n2k_config)} sections")
        
        model_config = loader.load("model_config")
        print(f"✓ model_config.yaml - {len(model_config)} sections")
        
        detection_config = loader.load("detection_config")
        print(f"✓ detection_config.yaml - {len(detection_config)} sections")
        
        print("\n✅ All configuration files valid!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}\n")
        return False


def test_directory_structure():
    """Test if all required directories exist"""
    print("="*60)
    print("TEST 4: Directory Structure")
    print("="*60)
    
    required_dirs = [
        'config',
        'data/raw/n2k/normal',
        'data/processed',
        'data/scalers',
        'data/thresholds',
        'src/initialization',
        'src/preprocessing',
        'src/models',
        'src/utils',
        'scripts',
        'models',
        'results',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required directories exist!\n")
    else:
        print("\n⚠️  Some directories missing - may need creation\n")
    
    return all_exist


def test_basic_functionality():
    """Test basic module functionality with dummy data"""
    print("="*60)
    print("TEST 5: Basic Functionality")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test signal selector
        from initialization.signal_selector import SignalSelector
        dummy_data = pd.DataFrame(np.random.rand(100, 10))
        selector = SignalSelector()
        signals = selector.select(dummy_data)
        print(f"✓ SignalSelector works - selected {len(signals)} signals")
        
        # Test FIFO queue
        from preprocessing.fifo_queue import FIFOQueue
        queue = FIFOQueue(max_size=100)
        queue.push(np.random.rand(10))
        queue.push(np.random.rand(10))
        print(f"✓ FIFOQueue works - size {len(queue)}")
        
        # Test view builder
        from preprocessing.view_builder import ViewBuilder
        builder = ViewBuilder()
        snapshot = np.random.rand(100, 10)
        view = builder.build_view(snapshot, period=5)
        print(f"✓ ViewBuilder works - view shape {view.shape}")
        
        # Test normalizer
        from preprocessing.normalizer import Normalizer
        normalizer = Normalizer()
        normalized = normalizer.fit_transform(dummy_data)
        print(f"✓ Normalizer works - normalized shape {normalized.shape}")
        
        # Test autoencoder
        from models.autoencoder import CNNAutoencoder
        ae = CNNAutoencoder(input_shape=(20, 10))
        ae.build()
        print(f"✓ CNNAutoencoder works - {len(ae.model.layers)} layers")
        
        print("\n✅ All modules function correctly!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "🧪 "*30)
    print("PROJECT VALIDATION - Testing Before Data Collection")
    print("🧪 "*30 + "\n")
    
    results = {
        'Imports': test_imports(),
        'Modules': test_project_modules(),
        'Configuration': test_configuration(),
        'Directories': test_directory_structure(),
        'Functionality': test_basic_functionality(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Your project is ready!")
        print("\n📋 Next steps:")
        print("  1. Generate dummy data: python scripts/00_generate_dummy_data.py")
        print("  2. Test pipeline with dummy data")
        print("  3. Fix any issues")
        print("  4. THEN start collecting real N2K data")
        print("\n⚠️  You DO NOT need attack data yet!")
        print("   - First: Train on normal data only")
        print("   - Later: Collect attack data for testing")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\n⚠️  Fix the issues above before proceeding")
        print("\n💡 Common fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check you're in project root directory")
        print("  - Ensure virtual environment is activated")
    
    print("\n")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
