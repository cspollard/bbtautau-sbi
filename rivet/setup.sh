export MGPATH=$HOME/Physics/mg5amcnlo-3.6.4
export RIVETPATH=$HOME/Physics/MCs
export PYTHIAPATH=$HOME/Physics/MCs/pythia8318
export RIVET_ANALYSIS_PATH=$PWD
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export MACOSX_DEPLOYMENT_TARGET=15.0
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$SDKROOT/usr/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

source $RIVETPATH/venv/rivetenv.sh
source $RIVETPATH/venv/yodaenv.sh
