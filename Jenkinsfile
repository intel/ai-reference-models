node() {
    stage('Style tests') {
        sh """
        #!/bin/bash -x
        set -e

        # install flake8 into a venv
        # get flake8 command not found otherwise...
        virtualenv -p python3 lintvenv
        . lintvenv/bin/activate

        pip install flake8
        flake8 benchmarks
        
        deactivate
        """
    }
    // put unit tests here later
    // stage('Unit tests') {
    //     echo 'Unit testing..'
    // }
    // put benchmarks here later
    // stage('Benchmarks') {
    //     echo 'Benchmark testing..'
    // }
}
