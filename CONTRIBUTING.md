# Contributing to EquiMed_DSS

Thank you for your interest in contributing to EquiMed_DSS! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/EquiMed_DSS.git
cd EquiMed_DSS

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Run tests to verify setup
pytest tests/
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix existing bugs in the codebase
- **New features**: Add new metrics or functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Add usage examples
- **Performance improvements**: Optimize existing code
- **Code refactoring**: Improve code quality

### Before You Start

1. Check existing issues and pull requests to avoid duplication
2. For major changes, open an issue first to discuss your proposal
3. Ensure your changes align with the project's goals

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) style guidelines with some modifications:

- **Line length**: Maximum 100 characters (not 79)
- **Imports**: Use `isort` to organize imports
- **Formatting**: Use `black` for code formatting
- **Type hints**: Use type hints for function signatures

### Code Formatting Tools

Run these tools before submitting:

```bash
# Format code with black
black equimed_dss

# Sort imports with isort
isort equimed_dss

# Check style with flake8
flake8 equimed_dss

# Type check with mypy
mypy equimed_dss
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `HierarchicalEquityRatio`)
- **Functions/Methods**: snake_case (e.g., `calculate_her`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_THRESHOLD`)
- **Private members**: prefix with underscore (e.g., `_internal_method`)

### Documentation Style

- Use Google-style docstrings
- Document all public classes, methods, and functions
- Include examples in docstrings where helpful

Example:

```python
def calculate_metric(data: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate the fairness metric for given data.

    Args:
        data: Input data array containing model predictions.
        threshold: Decision threshold for classification (default: 0.5).

    Returns:
        Dictionary containing metric values and interpretation.

    Raises:
        ValueError: If data is empty or invalid.

    Example:
        >>> data = np.array([0.8, 0.6, 0.9])
        >>> result = calculate_metric(data)
        >>> print(result['score'])
        0.85
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Aim for at least 80% code coverage
- Use descriptive test names
- Test edge cases and error conditions

### Test Structure

```python
import pytest
import numpy as np
from equimed_dss.domain2 import HierarchicalEquityRatio

class TestHierarchicalEquityRatio:
    """Test suite for HierarchicalEquityRatio metric."""

    def test_calculate_her_basic(self):
        """Test basic HER calculation with valid inputs."""
        metric = HierarchicalEquityRatio()
        scores = {'White': 0.85, 'Black': 0.75}
        result = metric.calculate_her(scores)

        assert 'White' in result
        assert 'Black' in result
        assert result['White']['score'] == 1.0

    def test_calculate_her_missing_reference(self):
        """Test HER calculation raises error when reference group missing."""
        metric = HierarchicalEquityRatio()
        scores = {'Black': 0.75, 'Hispanic': 0.80}

        with pytest.raises(ValueError):
            metric.calculate_her(scores, reference_group='White')
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=equimed_dss --cov-report=html

# Run specific test file
pytest tests/test_domain2.py

# Run specific test
pytest tests/test_domain2.py::TestHierarchicalEquityRatio::test_calculate_her_basic
```

## Pull Request Process

### Before Submitting

1. **Update your branch**: Rebase on the latest master
2. **Run tests**: Ensure all tests pass
3. **Run linters**: Fix all linting issues
4. **Update documentation**: Update docs if needed
5. **Add tests**: Include tests for new features

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Related Issues
Closes #123
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Reporting Bugs

### Before Reporting

- Check if the bug has already been reported
- Verify the bug exists in the latest version
- Collect information about your environment

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- EquiMed_DSS version: [e.g., 0.1.0]

## Additional Context
Any other relevant information
```

## Suggesting Enhancements

### Enhancement Proposal Template

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
What other approaches did you consider?

## Additional Context
Any other relevant information
```

## Questions?

If you have questions about contributing, please:
- Open an issue with the "question" label
- Contact the maintainers

## License

By contributing to EquiMed_DSS, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to EquiMed_DSS!
