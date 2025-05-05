# Bytropix Workspace Guide

## Repository Structure

## Development Workflow

### Setup Environment
1. Clone the repository
2. Run `setup.bat` to create a virtual environment and install dependencies
3. Activate the environment with `venv.bat`

### Training Workflows
- **Standard Training**: Use `runWuBuNest.bat` with appropriate parameters
- **Poetry Model**: Use `runWuBuNestPoem.bat` for poem structure generation
- **mRNA Sequences**: Use `runWuBuNestmRnaTrainer.bat` for nucleotide data

### Inference Usage
1. Train a model or obtain a pre-trained checkpoint
2. Use `WuBuNest_Inference.bat` for interactive generation
3. For programmatic usage, import from `WuBuNest_Inference.py`

### Data Preparation
1. Standard datasets: Use `setup.bat` (includes WikiText download)
2. Poetry datasets: Run `poem_dataset_generator.py`
3. Custom data: Use `convertdata.py` to convert to appropriate format

## Development Guidelines

### Code Organization
- Core geometric operations should go in `WuBuNesting_impl.py`
- Training logic belongs in specific trainer files
- Keep visualization separate in `wubu_nesting_visualization.py`
- Place experimental features in `draftPY/` directory

### Naming Conventions
- Class names: CamelCase
- Methods and functions: snake_case
- Constants: UPPER_SNAKE_CASE
- Module files: snake_case.py

### Documentation Standards
- Class and function docstrings: Google style
- Mathematical operations: Include LaTeX formulas in comments
- Complex algorithms: Add step-by-step explanations

### Testing
- Add tests for geometric operations in `tests/`
- Validate numeric stability across different parameter values
- Compare output against theoretical expectations

## Contribution Guidelines

### Adding Features
1. Start with an issue/ticket describing the feature
2. Create a branch with descriptive name
3. Implement the feature with appropriate tests
4. Update relevant documentation
5. Submit pull request for review

### Code Review Process
- All changes should be reviewed
- Must pass established tests
- Update documentation appropriately
- Check for performance regressions

### Performance Considerations
- Prefer vectorized operations
- Consider numerical stability in hyperbolic operations
- Test with both small and large batch sizes
- Monitor VRAM usage for training optimization
