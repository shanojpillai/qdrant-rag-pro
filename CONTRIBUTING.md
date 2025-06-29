# Contributing to QdrantRAG-Pro

Thank you for your interest in contributing to QdrantRAG-Pro! We welcome contributions from the community and are grateful for your help in making this project better.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git
- OpenAI API key for testing

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/qdrant-rag-pro.git
   cd qdrant-rag-pro
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

4. **Start development services**
   ```bash
   docker-compose up -d
   ```

5. **Run tests to ensure everything works**
   ```bash
   pytest tests/
   ```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **System information** (OS, Python version, etc.)
- **Relevant logs** or error messages
- **Screenshots** if applicable

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when creating issues.

### Feature Requests

For feature requests, please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** and problem you're trying to solve
- **Explain the proposed solution** or feature
- **Consider the scope** and impact on existing functionality

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

## 🔧 Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run these before submitting:

```bash
# Format code
black core/ scripts/ tests/

# Check linting
flake8 core/ scripts/ tests/

# Type checking
mypy core/

# Run tests
pytest tests/ -v
```

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(search): add hybrid search weighting
fix(embedding): handle empty text input
docs(readme): update installation instructions
```

### Testing

- Write tests for new features and bug fixes
- Ensure all tests pass before submitting
- Aim for good test coverage
- Use descriptive test names

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_search.py

# Run with coverage
pytest tests/ --cov=core/
```

### Documentation

- Update documentation for new features
- Include docstrings for all public functions and classes
- Add examples for complex functionality
- Update README if needed

## 📝 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run the full test suite
   pytest tests/
   
   # Check code style
   black --check core/ scripts/ tests/
   flake8 core/ scripts/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat(component): add new feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Use our [PR template](.github/pull_request_template.md)
   - Provide clear description of changes
   - Link related issues
   - Request review from maintainers

### PR Requirements

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts
- [ ] Linked to relevant issue(s)

## 🏗️ Project Structure

Understanding the project structure helps with contributions:

```
core/
├── config/          # Configuration management
├── database/        # Qdrant integration
├── services/        # Core business logic
└── models/          # Data models

scripts/             # Utility scripts
tests/              # Test suite
notebooks/          # Analysis notebooks
data/               # Sample data and outputs
```

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- **Performance optimizations** for search and embedding
- **Additional embedding providers** (Cohere, HuggingFace, etc.)
- **Enhanced filtering capabilities** for metadata
- **Monitoring and observability** improvements

### Medium Priority
- **Additional document formats** (PDF, DOCX, etc.)
- **API endpoints** for programmatic access
- **Caching improvements** for better performance
- **Documentation and tutorials**

### Good First Issues
- **Bug fixes** and small improvements
- **Test coverage** improvements
- **Documentation** updates and examples
- **Code cleanup** and refactoring

Look for issues labeled `good first issue` or `help wanted`.

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📞 Getting Help

If you need help:

- **Check the documentation** first
- **Search existing issues** for similar problems
- **Ask questions** in GitHub Discussions
- **Join our Discord** for real-time help

## 🙏 Recognition

Contributors will be:

- Listed in our [Contributors](CONTRIBUTORS.md) file
- Mentioned in release notes for significant contributions
- Invited to join our contributor Discord channel

Thank you for contributing to QdrantRAG-Pro! 🚀
