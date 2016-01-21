#!/usr/bin/env bash

# get the file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $DIR/../../src/
echo `pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`
# configure spark logging to WARN only
py.test -v deep_learning/test
popd