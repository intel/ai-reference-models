node('nervana-skx102.fm.intel.com') {
    // pull the code
    dir( 'intel-models' ) {
        checkout scm
    }
    stage('Style tests') {
        sh """
        #!/bin/bash -x
        set -e

        # install flake8 into a venv
        # get flake8 command not found otherwise...
        sudo easy_install virtualenv
        virtualenv -p python3 lintvenv
        . lintvenv/bin/activate

        pip install flake8
        flake8 intel-models/benchmarks
        
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
