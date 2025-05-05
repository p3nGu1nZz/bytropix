# Bytropix Repository Inventory

## Core Architecture

### WuBu Nesting Framework
- **WuBuNesting_impl.py**: Primary implementation of the WuBu Nesting architecture, including:
  - `HyperbolicUtils`: Core hyperbolic geometry operations (log/exp maps, gyrovector math)
  - `BoundaryManifold`: Learnable hyperbolic submanifolds implementation
  - `TangentSpaceRotation`: Manages rotations in tangent spaces
  - `InterLevelTransform`: Handles transitions between nested hyperbolic levels
  - `WuBuNestingLevel`: Implements single level processing
  - `WuBuNestingModel`: Orchestrates the full multi-level framework

### Sequence Models
- **WuBuNest_Trainer.py**: Training implementation for byte-level sequence modeling
  - Includes `WuBuNestingSequenceModel` class connecting WuBu to sequence tasks
  - Implements `HAKMEMLocalEncoder` and `HAKMEMLocalDecoder` for byte processing
  - Contains data handling classes for byte sequences
  - Integrates with hyperbolic architecture in tangent space
  
- **WuBuNestmRnaTrainer.py**: Specialized adaptation for nucleotide sequences
  - Modified for RNA/DNA sequence modeling using BioPython integration
  - Custom dataset classes for genomic data
  - Sequence-specific augmentations and processing

### Interference Models
- **bsfin_main.py**: Main implementation of BSFIN architecture
  - Babylon Index Semantic Field Interference Network
  - Complex-valued representations for quantum-inspired modeling
  - Entangled interference layers for non-linear interactions
  
- **sfin_inference.py**: Inference script for BSFIN models
  - Implements interactive generation with specialized settings
  - Contains quantum noise injection options

### Hyperbolic Category Discovery
- **HypCD.py**: Implementation for hierarchical category learning
  - Hyperbolic embeddings for hierarchical relationships
  - Optimization in hyperbolic space
  - Based on research in hierarchical structure discovery

## Mathematical Foundation

### Hyperbolic Geometry Components
- `PoincareBall`: Implements the Poincaré ball model of hyperbolic geometry
- `Manifold`: Abstract base class for differential geometry operations
- `SO_n_Rotation`: Special Orthogonal group rotations for tangent spaces
- `GyroLinear`: Gyrovector-based linear mappings in hyperbolic space
- `RiemannianLayerNorm`: Layer normalization adapted for hyperbolic space
- `HyperbolicDistanceAttention`: Attention mechanism using hyperbolic distances

### Quaternion Operations
- `hamilton_product`: Implements quaternion multiplication
- `quat_conjugate`: Computes quaternion conjugates
- `quat_rotate_via_pvq`: Rotation via quaternion sandwich product
- `QuaternionLinear`: Linear transformation using quaternion algebra

### HAKMEM Components
- `HAKMEMEntropyHelper`: Entropy calculations inspired by HAKMEM memo
- `HAKMEMBabylonIndex`: Patching mechanism for byte sequences
- `HAKMEMQController`: Q-learning controller for hyperparameter tuning
- `HAKMEMCrossAttentionBlock`: Enhanced cross-attention implementation

## Training and Infrastructure

### Optimization Components
- **EnhancedSGD.py**: Custom optimizer with Q-learning capabilities
  - Adaptive learning rate and momentum adjustment
  - Gradient statistics tracking
  - Parameter group handling with different learning rates

### Data Processing
- **convertdata.py**: Utilities for data transformation including:
  - Functions for processing various text datasets
  - Conversion between formats
  - Sampling and preprocessing

- **poem_dataset_generator.py**: Specialized dataset creation for poetry
  - Structure-focused synthetic data generation
  - Poetry format templates 
  - Training/validation split creation

### Execution Scripts
- **runWuBuNest.bat**: Main training execution script with parameters
- **runWuBuNestPoem.bat**: Poetry-specific training script
- **runWuBuNestmRnaTrainer.bat**: mRNA/DNA sequence training
- **WuBuNest_Inference.bat**: Interactive inference execution
- **setup.bat**: Environment setup and dependency installation
- Multiple draft execution scripts in draftPY directory

## Visualization and Examples

- **wubu_nesting_visualization.py**: Visualization tools including:
  - `visualize_poincare_disk`: 2D visualization of Poincaré disk embeddings
  - `visualize_poincare_nested`: Multi-level nesting visualization
  - `visualize_tangent_space_transition`: Transition operations visualization

- **wubu_nesting_example.py**: Example usage demonstrations
  - Synthetic hierarchical data generation
  - Model instantiation and training examples
  - Evaluation and result visualization

## Documentation

- **WuBuHypCD.tex**: LaTeX document with theoretical foundation
- **WuBuHypCD-paper.md**: Markdown version of the theoretical paper
- **README.md**: Project overview and basic usage
- Various documentation in the docs directory

## Dependencies and Configuration

- **requirements.txt**: Core Python dependencies
- **.gitignore**: Version control exclusions
- **references.bib**: Bibliography for academic references