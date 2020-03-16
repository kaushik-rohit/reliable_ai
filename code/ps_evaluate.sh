#!/bin/bash

for net in fc1 fc2 fc3 fc4 fc5 conv1 conv2 conv3 conv4 conv5
do
	echo Evaluating network ${net}...
	# shellcheck disable=SC2045
	# shellcheck disable=SC2006
	for spec in `ls ../ps_test_cases/${net}`
	do
		python verifier.py --net ${net} --spec ../ps_test_cases/${net}/"${spec}"
	done
done
