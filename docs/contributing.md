# Contributing to Mukh

Thank you for your interest in contributing to Mukh! This guide will help you get started with contributing to the project.

## Getting Started

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mukh.git
   cd mukh
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e .
   pip install -r requirements_dev.txt
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Project Structure

```
mukh/
├── mukh/                    # Main package
│   ├── core/               # Core functionality
│   ├── face_detection/     # Face detection models
│   ├── deepfake_detection/ # Deepfake detection
│   ├── reenactment/        # Face reenactment
│   └── utils/              # Utility functions
├── examples/               # Usage examples
├── tests/                  # Test suite
├── docs/                   # Documentation
├── assets/                 # Test assets
└── setup.py               # Package configuration
```

## Development Guidelines

### Code Style

- **Python Version**: Python 3.7+
- **Code Formatting**: Use `black` for code formatting
- **Import Sorting**: Use `isort` for import organization
- **Linting**: Use `flake8` for code linting
- **Type Hints**: Use type hints where possible

### Documentation Style

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Examples**: Include usage examples in docstrings
- **API Documentation**: All public APIs must be documented

Example docstring format:
```python
def detect_faces(image_path: str, confidence_threshold: float = 0.5) -> List[Detection]:
    """
    Detect faces in an image.
    
    Args:
        image_path (str): Path to the input image file.
        confidence_threshold (float): Minimum confidence for face detection.
            Defaults to 0.5.
    
    Returns:
        List[Detection]: List of face detections with bounding boxes and confidence scores.
    
    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If confidence_threshold is not between 0 and 1.
    
    Example:
        ```python
        from mukh.face_detection import FaceDetector
        
        detector = FaceDetector.create("mediapipe")
        detections = detector.detect("path/to/image.jpg")
        print(f"Found {len(detections)} faces")
        ```
    """
```

### Testing

- **Test Coverage**: Aim for >90% test coverage
- **Test Framework**: Use `pytest` for testing
- **Test Organization**: Mirror the package structure in tests
- **Mock External Dependencies**: Use `pytest-mock` for mocking

Running tests:
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=mukh

# Run specific test file
pytest tests/test_face_detection.py

# Run tests with specific marker
pytest -m "not slow"
```

### Git Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new face detection model"
   ```

4. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Commit Message Convention

Use conventional commits format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `style:` for formatting changes
- `ci:` for CI/CD changes

## Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Code snippets or error messages

### 💡 Feature Requests

For feature requests, please provide:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Examples of similar features in other libraries

### 🔧 Code Contributions

#### Adding New Models

To add a new face detection model:

1. **Create Model Class**
   ```python
   # mukh/face_detection/models/your_model.py
   from mukh.core.base import BaseDetector
   
   class YourModelDetector(BaseDetector):
       """Your model detector implementation."""
       
       def __init__(self, confidence_threshold: float = 0.5):
           """Initialize the detector."""
           super().__init__(confidence_threshold)
           # Initialize your model here
       
       def detect(self, image_path: str) -> List[Detection]:
           """Detect faces in an image."""
           # Implement detection logic
           pass
   ```

2. **Register Model**
   ```python
   # mukh/face_detection/__init__.py
   from .models.your_model import YourModelDetector
   
   AVAILABLE_MODELS = {
       # ... existing models
       "your_model": YourModelDetector,
   }
   ```

3. **Add Tests**
   ```python
   # tests/face_detection/test_your_model.py
   import pytest
   from mukh.face_detection import FaceDetector
   
   def test_your_model_creation():
       detector = FaceDetector.create("your_model")
       assert detector is not None
   
   def test_your_model_detection():
       detector = FaceDetector.create("your_model")
       detections = detector.detect("assets/test_image.jpg")
       assert isinstance(detections, list)
   ```

4. **Update Documentation**
   - Add model to API documentation
   - Include usage examples
   - Update model comparison tables

#### Adding New Features

1. **Design the API**
   - Follow existing patterns
   - Ensure consistency with other modules
   - Consider backward compatibility

2. **Implement the Feature**
   - Write clean, documented code
   - Handle edge cases and errors
   - Follow performance best practices

3. **Add Comprehensive Tests**
   - Unit tests for individual functions
   - Integration tests for end-to-end workflows
   - Performance tests for computationally intensive features

4. **Update Documentation**
   - API reference documentation
   - Usage examples
   - Tutorial or guide if applicable

### 📚 Documentation Contributions

- **API Documentation**: Improve docstrings and API references
- **Tutorials**: Create step-by-step guides for common use cases
- **Examples**: Add practical examples and use cases
- **README**: Improve project description and quick start guide

### 🧪 Testing Contributions

- **Increase Test Coverage**: Add tests for uncovered code paths
- **Performance Tests**: Add benchmarks and performance tests
- **Integration Tests**: Test interactions between components
- **CI/CD Improvements**: Enhance automated testing workflows

## Code Review Process

### For Contributors

1. **Self-Review**: Review your own code before submitting
2. **Description**: Provide clear PR description with context
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update relevant documentation

### For Reviewers

1. **Be Constructive**: Provide helpful feedback
2. **Check Functionality**: Verify the feature works as intended
3. **Review Tests**: Ensure adequate test coverage
4. **Documentation Review**: Check for clear documentation

## Release Process

Mukh follows semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Community Guidelines

### Code of Conduct

- **Be Respectful**: Treat all contributors with respect
- **Be Inclusive**: Welcome contributions from everyone
- **Be Collaborative**: Work together to improve the project
- **Be Constructive**: Provide helpful feedback and suggestions

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussions
- **Documentation**: Check existing documentation first
- **Examples**: Look at examples for common use cases

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted in releases
- **Documentation**: Contributors credited in relevant sections

## Development Tools

### Recommended IDE Setup

- **VS Code**: With Python extension
- **PyCharm**: Professional or Community edition
- **Vim/Neovim**: With appropriate Python plugins

### Useful Tools

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **pre-commit**: Git hooks

### Environment Variables

Set these for development:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MUKH_DEBUG=1  # Enable debug logging
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure package is installed in development mode
2. **Test Failures**: Check if all dependencies are installed
3. **Pre-commit Failures**: Run `black` and `isort` manually
4. **Documentation Build**: Ensure all dependencies in `requirements_dev.txt`

### Getting Unstuck

If you're stuck:
1. Check existing issues and discussions
2. Read the documentation thoroughly
3. Look at similar implementations in the codebase
4. Ask for help in GitHub discussions

## Thank You!

Your contributions make Mukh better for everyone. Whether you're fixing bugs, adding features, improving documentation, or helping other users, every contribution is valuable and appreciated!

For questions about contributing, feel free to open an issue or start a discussion on GitHub. 