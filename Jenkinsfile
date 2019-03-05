node('skx') {
    try {
        // simulate the test not having label, ie permission, to run
        noPermissions = false
        if (true) {
            noPermissions = true
            error('Aborting the build, no permissions found.')
        }

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
                sudo apt-get install -y python3-dev || true
                sudo yum install -y python36-devel.x86_64 || true

                # virtualenv 16.3.0 is broken do not use it
                sudo python2 -m pip install --upgrade pip virtualenv!=16.3.0 tox
                sudo python3 -m pip install --upgrade pip virtualenv!=16.3.0 tox
                """
            }
            stage('Style tests') {
                sh """
                #!/bin/bash -x
                set -e

                cd intel-models
                tox -e py2.7-flake8 -e py3-flake8
                """
            }
            stage('Unit tests') {
                sh """
                #!/bin/bash -x
                set -e

                cd intel-models
                tox -e py2.7-py.test -e py3-py.test
                """
            }
            // put benchmarks here later
            // stage('Benchmarks') {
            //     echo 'Benchmark testing..'
            // }
        }
    } catch (e) {
        if (noPermissions) {
            currentBuild.result = 'ABORTED'
            echo('Label "test me" not applied to this PR')
            // return here instead of throwing error to keep the build "green"
            return
        }
        // If there was an exception thrown, the build failed
        currentBuild.result = "FAILURE"
        throw e
    } finally {
    } // finally
}
