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
        stage('Style tests') {
            sh """
            #!/bin/bash -x
            set -e
            sudo easy_install virtualenv

            cd intel-models

            make lint
            """
        }
        stage('Unit tests') {
            sh """
            #!/bin/bash -x
            set -e

            cd intel-models

            make unit_test
            make unit_test3
            """
        }
        // put benchmarks here later
        // stage('Benchmarks') {
        //     echo 'Benchmark testing..'
        // }
    }
}
