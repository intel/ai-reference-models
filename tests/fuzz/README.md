## Fuzz Testing in Intel® AI Reference Models
Fuzz testing is an automated software testing technique that involves providing invalid, unexpected, or random data as inputs to a computer program. The program is monitored for exceptions, such as crashes and failing assertions. These test files use Atheris, a coverage-guided Python fuzzing engine, to conduct fuzz testing in AI Reference Models. 

### Requirements
* Python: Version 3.9 or newer
* Atheris: Google's fuzzing engine for Python
* Coverage: Code coverage measurement for Python

## Setup
To prepare your environment for fuzz testing with Atheris, follow these steps:

# Install Test Dependencies
```
pip install -r ../../requirements.txt
```
## Running Fuzz Tests
Example 1 (runs until interrupted by the user or an exception is thrown):
```
python3 -m coverage run fuzz_validators.py -atheris_runs=0
```

Example 2 (runs for 10000 iterations and adds to coverage report instead of overwriting):
```
python3 -m coverage run -a fuzz_validators.py -atheris_runs=10000
```
# Interpreting Results
When running fuzz tests, it is important to understand the output to identify potential issues effectively.

### Crashes and Exceptions
Atheris reports when the fuzzed input causes the program to crash or raise unhandled exceptions. These are crucial for identifying vulnerabilities. 

### Coverage Metrics
Fuzzing code coverage is captured, which helps in understanding which parts of the code were exercised. Low coverage might indicate that additional fuzzing targets or more diverse inputs are needed. 

To generate the coverage report after running a fuzz test, run the following command inside the fuzz folder:
```
python3 -m coverage report
```

The output will look like:

| Name                                      | Stmts | Miss | Cover |
|-------------------------------------------|-------|------|-------|
| ../benchmarks/common/__init__.py          | 0     | 0    | 100%  |
| ../benchmarks/common/utils/__init__.py    | 0     | 0    | 100%  |
| ../benchmarks/common/utils/validators.py  | 64    | 3    |  95%  |
| fuzz_validators.py                        | 24    | 0    | 100%  |
|-------------------------------------------|-------|------|-------|
| TOTAL                                     | 88    | 3    | 97%   |

The test may not be designed to exercise all the instrumented code, only a certain 
part of it. It can be more helpful to look at the individual file coverage than the total. 

The coverage report can also be viewed interactively, to inspect files or functions executed, using html:
```
python3 -m coverage html
cd htmlcov
python3 -m http.server
```

Then, open http://localhost:8000/index.html in a web browser.

### Reproducing Issues 
When a failure is encountered, Atheris outputs a test case that can reproduce the issue.
These test cases can help you debug and fix the vulnerabilities in the code.
Here is an example of the output you might see:
```
==61244== ERROR: libFuzzer: fuzz target exited
SUMMARY: libFuzzer: fuzz target exited
MS: 5 ChangeBinInt-CopyPart-CrossOver-EraseBytes-InsertByte-; base unit: adc83b19e793491b1c6ea0fd8b46cd9f32e592fc
0x5d,0xa,0x0,
]\012\000
artifact_prefix='./'; Test unit written to ./crash-335f7afdb5718050a47e623343be45a5ee1e5f17
Base64: XQoA
```

To reproduce, run:
```
python3 -m coverage run fuzz_validators.py ./crash-335f7afdb5718050a47e623343be45a5ee1e5f17
```
