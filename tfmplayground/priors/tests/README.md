# Priors Module Unit Tests

## Overview

Comprehensive unit tests for the `tfmplayground.priors` module, covering all core functionality including configuration, data loading, and utility functions.

## Running the Tests

### Option 1: Using unittest

```bash
# Run all tests
python -m unittest discover -s tfmplayground/priors/tests -p "test_*.py" -v

# or

python run_all_tests.py

# Run specific test file
python -m unittest tfmplayground.priors.tests.test_config -v

# Run specific test class
python -m unittest tfmplayground.priors.tests.test_config.TestGetTICLPriorConfig -v
```

### Option 2: Individual test files

```bash
# Test config
python tfmplayground/priors/tests/test_config.py

# Test utils
python tfmplayground/priors/tests/test_utils.py

# Test dataloaders
python tfmplayground/priors/tests/test_dataloader.py
```

## Test Features

### Mock Data Generation
Tests create mock data and temporary files to avoid dependencies on external data sources.

### Cleanup
All tests properly clean up temporary files and directories after execution.

### Isolation
Each test is independent and doesn't affect others.

### Edge Cases
Tests cover edge cases like:
- Empty batches
- Pointer wraparound in file loaders
- Tensor vs scalar single_eval_pos
- Data padding and truncation
- Invalid configurations

## Adding New Tests

When adding new functionality to the priors module:

1. Create test methods following the naming convention `test_<feature_name>`
2. Use descriptive docstrings
3. Include both positive and negative test cases
4. Clean up any resources (files, etc.) in tearDown
5. Run existing tests to ensure no regressions

### Example Test Template

```python
def test_new_feature(self):
    """Test description of what this validates."""
    # Setup
    input_data = create_test_data()
    
    # Execute
    result = function_under_test(input_data)
    
    # Assert
    self.assertEqual(result, expected_value)
    self.assertIsNotNone(result)
```