# Running Tests

## Prerequisites

Running tests require the following dependencies to be installed:
* [python](https://www.python.org/downloads/) 2.7 or newer
* python developer libraries (`python-dev` for Ubuntu or `python-devel` for CentOS 7)
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

To run style checks using python 2:
```
make lint2
```

To run style checks using python 3:
```
make lint3
```

To run style checks using both python 2 and python 3:
```
make lint
```

## Running only unit tests

Unit tests can be run using the commands below.

To run unit tests using python 2:
```
make unit_test2
```

To run unit tests using python 3:
```
make unit_test3
```

To run unit tests using python 2 and python 3:
```
make unit_test
```

## Running individual unit tests

To run individual unit tests using python 2:

- Run one complete test file:
  ```
  tox -e py2.7-py.test -- <test_file_path>
  ```
- Run one test in a file: 
  ```
  tox -e py2.7-py.test -- <test_file_path>::<test_name>
  ```
- Run all tests containing a substring:
  ```
  tox -e py2.7-py.test -- -k <substring>
  ```

To run individual unit tests using python 3:

- Run one complete test file:
  ```
  tox -e py3-py.test -- <test_file_path>
  ```
- Run one test in a file: 
  ```
  tox -e py3-py.test -- <test_file_path>::<test_name>
  ```
- Run all tests containing a substring:
  ```
  tox -e py3-py.test -- -k <substring>
  ```