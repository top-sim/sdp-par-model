#!/usr/bin/env groovy

pipeline {
    agent {label 'sdp-ci-01'}
    environment {
        MPLBACKEND='agg'
    }
    stages {
        stage('Setup') {
            steps {
                sh '''
# Set up fresh Python virtual environment
virtualenv -p `which python3` --no-site-packages _build
. _build/bin/activate
export MPLBACKEND=agg

# Install requirements
pip install -U pip setuptools
pip install -r requirements.txt
pip install pymp-pypi
jupyter nbextension enable --py widgetsnbextension
pip install pytest pytest-xdist pytest-cov
'''
            }
        }
        stage('Test') {
            steps {
                sh '''
cd $WORKSPACE
. _build/bin/activate
export MPLBACKEND=agg

py.test -n 4 --verbose tests
'''
            }
        }

       stage('Run Notebooks') {
           steps {
               sh '''
cd $WORKSPACE
. $WORKSPACE/_build/bin/activate
export MPLBACKEND=agg

mkdir $WORKSPACE/out || true
for i in $WORKSPACE/iPython/*.ipynb ; do
 (jupyter nbconvert --execute --ExecutePreprocessor.timeout=4500 --to html --log-level=20 --output-dir=$WORKSPACE/out $i || echo $i >> failed)
done
wait
if [ -f failed ]; then
 echo Export of following notebooks failed:
 cat failed
 exit 1
fi

cp -R $WORKSPACE/iPython/out $WORKSPACE/out || true
cp -R $WORKSPACE/compare_* $WORKSPACE/out || true
if [ -d $WORKSPACE/iPython/graphs ]; then
   cp -R $WORKSPACE/iPython/graphs $WORKSPACE/out
fi
'''
           }
       }
        stage ('Publish Results') {
            when { branch 'master'}
            steps {
                sshPublisher alwaysPublishFromMaster: true,
                publishers: [sshPublisherDesc(configName: 'vm12',
                                transfers: [sshTransfer(excludes: '',
                                        execCommand: '', execTimeout: 120000,
                                        flatten: false,
                                        makeEmptyDirs: false,
                                        noDefaultExcludes: false,
                                        patternSeparator: '[, ]+',
                                        remoteDirectory: 'sdp-par-model',
                                        remoteDirectorySDF: false,
                                        removePrefix: '',
                                        sourceFiles: 'out/**')],
                                usePromotionTimestamp: false,
                                useWorkspaceInPromotion: false,
                                verbose: false)]
            }
        }
    }
    post {
        failure {
            emailext attachLog: true, body: '$DEFAULT_CONTENT',
                recipientProviders: [culprits()],
                subject: '$DEFAULT_SUBJECT',
                to: '$DEFAULT_RECIPIENTS'
        }
        fixed {
            emailext body: '$DEFAULT_CONTENT',
                recipientProviders: [culprits()],
                subject: '$DEFAULT_SUBJECT',
                to: '$DEFAULT_RECIPIENTS'
        }
    }
}
