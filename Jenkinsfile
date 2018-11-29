node('skx') {
    // Create a workspace path.  We need this to be 79 chars max, otherwise some nodes fail.
    // The workspace path varies by node so get that path, and then add on 10 chars of a UUID string.
    ws_path = "$WORKSPACE".substring(0, "$WORKSPACE".indexOf("workspace/") + "workspace/".length()) + UUID.randomUUID().toString().substring(0, 10)
    ws(ws_path) {
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

            pip install -r intel-models/tests/requirements.txt
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
}
