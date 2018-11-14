node() {
    stage('Style tests') {
        sh """
        #!/bin/bash -x
        set -e

        pip install flake8
        flake8 benchmarks
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
