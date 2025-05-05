# Bytropix Technical Specification

## 1. Core Architecture Overview

### 1.1 WuBu Nesting Framework
Bytropix implements the "WuBu Nesting" (層疊嵌套 - céngdié qiàntào) architecture, which creates a recursive hierarchy of hyperbolic spaces with adaptive geometry. Each level `i` operates in a Poincaré ball manifold `H^n_i_{c_i, s_i}` where:
- `n_i`: Dimensionality of the hyperbolic space (configurable per level)
- `c_i`: Curvature parameter (learnable, constrained to be positive)
- `s_i`: Scale parameter (learnable, controls the relative size)

### 1.2 Information Flow
Data flows through the multi-level architecture via tangent space transitions:
1. Input byte sequences → Babylon Index Patching → Meaningful segments
2. Segments → Local Encoder → Initial embeddings
3. Embeddings → Hyperbolic projection → Processing through WuBu Nesting levels
4. Multi-level representation → Aggregation → Output prediction

### 1.3 Hyperbolic Operations
The system implements the following key operations:
- **Exponential map**: Maps from tangent space to hyperbolic space
- **Logarithmic map**: Maps from hyperbolic space to tangent space
- **Gyrovector operations**: For computing in hyperbolic space
- **Inter-level transitions**: Transfers information between nested levels

## 2. Component Specifications

### 2.1 Bytropix Base Model
- **Input**: Raw UTF-8 bytes (vocabulary-free)
- **Patch Creation**: Entropy-based segmentation via BabylonIndex
- **Context Window**: Configurable, default 256 bytes
- **Output**: Byte-level prediction over 256 possible values

### 2.2 WuBu Nesting Modules
- **Boundary Manifolds**: Learnable points representing substructures at each level
- **Inter-Level Transforms**: Coordinate transformations between levels via tangent spaces
- **Tangent Flows**: Intra-level dynamics within tangent spaces
- **Level Descriptors**: Vectors capturing level-specific context
- **Spread Parameters**: Uncertainty or density parameters passed between levels

### 2.3 Geometric Parameters
- **Number of Levels**: Configurable, typically 3-4
- **Hyperbolic Dimensions**: Typically decreasing (e.g., 128→64→32)
- **Initial Curvatures**: Typically starting at 1.0, learnable during training
- **Boundary Points**: Configurable per level (e.g., 5→4→3)

## 3. Training Configuration

### 3.1 Training Process
- **Optimizer**: RiemannianEnhancedSGD with Q-learning parameter tuning
- **Batch Size**: Configurable, typically 16-32 depending on model size
- **Gradient Accumulation**: Supported for effective batch size manipulation
- **Mixed Precision**: Optional AMP support for training efficiency

### 3.2 Adaptation Options
- **mRNA Variant**: Specialized for nucleotide sequences
- **Poetry Structure Model**: Optimized for capturing poetic forms
- **Document Processing**: Enhanced with features for document structure

## 4. Inference Specifications

### 4.1 Generation Parameters
- **Temperature**: Controls randomness (0.3-0.9 typical range)
- **Repetition Penalty**: Discourages repetitive output (1.1-1.3 typical)
- **Top-k/Top-p**: Supported for controlling output distribution
- **Entropy Thresholds**: Customizable for controlling attention weights

### 4.2 Deployment Options
- **Interactive Mode**: CLI interface for real-time generation
- **Batch Processing**: Script support for bulk inference
- **Model Quantization**: Not currently implemented, planned for future

## 5. Technical Requirements

### 5.1 Hardware Requirements
- **Training**: CUDA-compatible GPU with 8GB+ VRAM recommended
- **Inference**: CPU-capable, GPU recommended for interactive use
- **RAM**: Minimum 8GB, 16GB+ recommended for larger models

### 5.2 Software Dependencies
- **Python**: 3.8+
- **PyTorch**: 1.10+
- **Key Libraries**: geoopt, wandb (optional), matplotlib (visualization)