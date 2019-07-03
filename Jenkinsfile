node('skx') {
    deleteDir()
    // Create a workspace path.  We need this to be 79 chars max, otherwise some nodes fail.
    // The workspace path varies by node so get that path, and then add on 10 chars of a UUID string.
    ws_path = "$WORKSPACE".substring(0, "$WORKSPACE".indexOf("workspace/") + "workspace/".length()) + UUID.randomUUID().toString().substring(0, 10)
    ws(ws_path) {
        // pull the code
        dir( 'intel-models' ) {
            checkout scm
        }
        stage('Install dependencies') {
            sh """
            #!/bin/bash -x
            set -e
            # don't know OS, so trying both apt-get and yum install
            sudo apt-get install -y python3-dev || sudo yum install -y python36-devel.x86_64

            # virtualenv 16.3.0 is broken do not use it
            python2 -m pip install --no-cache-dir --user --upgrade pip==19.0.3 virtualenv!=16.3.0 tox
            python3 -m pip install --no-cache-dir --user --upgrade pip==19.0.3 virtualenv!=16.3.0 tox
            """
        }
        stage('Style tests') {
            sh """
            #!/bin/bash -x
            set -e

            cd intel-models
            ~/.local/bin/tox -e py2.7-flake8 -e py3-flake8
            """
        }
        stage('Unit tests') {
            sh """
            #!/bin/bash -x
            set -e

            cd intel-models
            ~/.local/bin/tox -e py2.7-py.test -e py3-py.test
            """
        }
        // put benchmarks here later
        // stage('Benchmarks') {
        //     echo 'Benchmark testing..'
        // }
    }
}
