# Contributing to temporalcv

Thank you for your interest in contributing to temporalcv!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/temporalcv.git
cd temporalcv

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=temporalcv --cov-report=term-missing

# Run type checking
mypy src/temporalcv
```

## Code Style

- We use **ruff** for linting and formatting (black-compatible)
- Type hints are required on all public functions
- Docstrings follow NumPy style

Pre-commit hooks will automatically check these before each commit.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

## Commit Messages

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.
