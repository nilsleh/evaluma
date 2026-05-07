# Contributing

Contributions are welcome. This page covers how to set up a development environment, the coding practices the project follows, and how to build the documentation locally.

## Setting up

```bash
git clone https://github.com/nilsleh/evaluma
cd evaluma
pip install -e ".[dev]"
```

Run the test suite to confirm everything works:

```bash
pytest
```

## Workflow

1. Open an issue describing the bug or feature before writing code.
2. Fork the repository and create a branch from `main`.
3. Write tests before implementing (the project follows a TDD approach — new behaviour should be covered before it is written).
4. Open a pull request against `main`. The CI pipeline must be green before merging.


## Running the tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=evaluma --cov-report=term-missing

# Run a single file
pytest tests/test_iqm.py -v
```

The project targets 100% coverage of the `evaluma/` package. New code must be accompanied by tests that exercise it.

## Building the documentation locally

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build once and open the result in a browser:

```bash
make clean
make html
```

Open the build `index.html` file in your browser.
