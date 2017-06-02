pipeline {
    agent any

    environment {
       CXX = "g++-4.9.4"
       LD = "g++-4.9.4"
       ETL_MKL = 'true'
    }

    stages {
        stage ('git'){
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]],
                    submoduleCfg: [],
                    userRemoteConfigs: scm.userRemoteConfigs])
            }
        }

        stage ('pre-analysis') {
            steps {
                sh 'cppcheck --xml-version=2 -j3 --enable=all --std=c++11 `git ls-files "*.hpp" "*.cpp"` 2> cppcheck_report.xml'
                sh 'sloccount --duplicates --wide --details include/etl test workbench > sloccount.sc'
                sh 'cccc include/etl/*.hpp test/*.cpp workbench/*.cpp || true'
            }
        }

        stage ('build'){
            steps {
                sh 'make clean'
                sh 'make -j6 release'
            }
        }

        stage ('test'){
            steps {
                sh 'ETL_THREADS=-j6 ETL_GPP=g++-4.9.4 LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}:/opt/intel/mkl/lib/intel64:/opt/intel/lib/intel64\" ./scripts/test_runner.sh'
                archive 'catch_report.xml'
                junit 'catch_report.xml'
            }
        }

        stage ('sonar-master'){
            when {
                branch 'master'
            }
            steps {
                sh "/opt/sonar-runner/bin/sonar-runner"
            }
        }

        stage ('sonar-branch'){
            when {
                not {
                    branch 'master'
                }
            }
            steps {
                sh "/opt/sonar-runner/bin/sonar-runner -Dsonar.branch=${env.BRANCH_NAME}"
            }
        }

        stage ('bench'){
            steps {
                build job: 'etl - benchmark', wait: false
            }
        }
    }

    post {
        always {
            step([$class: 'Mailer',
                notifyEveryUnstableBuild: true,
                recipients: "baptiste.wicht@gmail.com",
                sendToIndividuals: true])
        }
    }
}
