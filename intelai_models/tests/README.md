# Running Tests

## Prerequisites

Running tests require the following dependencies to be installed:
* [python](https://www.python.org/downloads/) 3.6 or newer
* python developer libraries (`python3-dev` for Ubuntu 18.04 or `python36-devel` for CentOS 7)
* [tox](https://tox.readthedocs.io/en/latest/install.html) 3.7 or newer
* [virtualenv](https://virtualenv.pypa.io)
* [requirements-test.txt](../requirements-test.txt)

## Running all tests

To run style checks and unit tests for both python 2 and python 3:

```
make test
```

## Running only style checks

The following commands run flake8 style checks on the `/benchmarks` directory.

To run style checks using python 3:
```
make lint
```

## Running only unit tests

Unit tests can be run using the commands below.

```
make unit_test3
```

To run unit tests using python 3:
```
make unit_test
```

## Running a selection of unit tests

- Run one complete test directory or file:
  ```
  TESTFILES=<test_file_path> tox -e py3-py.test
  ```
- Run one test in a file:
  ```
  TESTFILES=<test_file_path>::<test_name> tox -e py3-py.test
  ```
- Run all tests containing a substring:
  ```
  TESTOPTS="-k <substring> --no-cov" tox -e py3-py.test
  ```
