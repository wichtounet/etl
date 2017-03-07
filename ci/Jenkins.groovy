node {
    try {
       stage 'git'
       checkout([$class: 'GitSCM', branches: [[name: '*/master']], doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]], submoduleCfg: [], userRemoteConfigs: [[url: 'https://github.com/wichtounet/etl.git']]])

       stage 'pre-analysis'
       sh 'cppcheck --xml-version=2 -j3 --enable=all --std=c++11 `git ls-files "*.hpp" "*.cpp"` 2> cppcheck_report.xml'
       sh 'sloccount --duplicates --wide --details include/etl test workbench > sloccount.sc'
       sh 'cccc include/etl/*.hpp test/*.cpp workbench/*.cpp || true'

       env.ETL_THREADS="-j6"
       env.CXX="g++-4.9.4"
       env.LD="g++-4.9.4"
       env.ETL_GPP="g++-4.9.4"
       env.LD_LIBRARY_PATH="${env.LD_LIBRARY_PATH}:/opt/intel/mkl/lib/intel64"
       env.LD_LIBRARY_PATH="${env.LD_LIBRARY_PATH}:/opt/intel/lib/intel64"

       stage 'build'
       sh 'make clean'
       sh 'make -j6 release'

       stage 'test'
       sh './scripts/test_runner.sh'

       stage 'sonar'
       sh '/opt/sonar-runner/bin/sonar-runner'

       stage 'bench'
       build job: 'etl - benchmark', wait: false
       currentBuild.result = 'SUCCESS'
    } catch (any) {
       currentBuild.result = 'FAILURE'
       throw any
   } finally {
       step([$class: 'Mailer',
           notifyEveryUnstableBuild: true,
           recipients: "baptiste.wicht@gmail.com",
           sendToIndividuals: true])
   }
}
