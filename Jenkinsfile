#!groovy

pipeline {
    agent any

    stages {
        stage ('git'){
            steps {
                // TODO This horrible Jenkins mess should be cleaned
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
            environment {
               CXX = "g++-6.4.0"
               LD = "g++-6.4.0"
               ETL_MKL = 'true'
               ETL_COVERAGE = 'true'
            }

            steps {
                sh 'make clean'
                sh 'make -j6 release'
            }
        }

        stage ('test'){
            environment {
                ETL_THREADS = "-j6"
                ETL_GPP = "g++-6.4.0"
            }

            steps {
                sh './scripts/test_runner_ci.sh'
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
            script {
                if (currentBuild.result == null) {
                    currentBuild.result = 'SUCCESS'
                }
            }

            step([$class: 'Mailer',
                notifyEveryUnstableBuild: true,
                recipients: "baptiste.wicht@gmail.com",
                sendToIndividuals: true])
        }
    }
}
