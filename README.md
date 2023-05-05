# OpenSearch ML-Commons Sanity Check
Sanity check script in python for OpenSearch ML-Commons plugin

## Prerequisite
* An OpenSearch cluster with security plugin enabled
* Python 3 installed (ideally 3.8)
* requests package

## Usage
python3 san_check.py host port [--working_directory [folder with your import data and its index]] [--auth account password] [--ml_node_only] [--memory_cb_activate]
* Host and port argument are __always required__ and should be __separated by space__. Separating by colon(:) is not accepted. The host address should __explicitly specify the protocol__ (We only support https now)
* --working_directory can be shortened as -wd, when this argument is not specified, data import process won't be activated.
* --ml_node_only can beshortened as -ML. If __not__ specified, the ml commons plugin can be run on all nodes.
* --memory_cb_activate can be shortened as -CB. If __not__ specified, the circuit breaker is deactivated.

## TODO
* Support non-secure mode
* Return an error counter when test is finished
* Support PPL command test
* Support sanity test on Custom Model and Neural Search
* Support sanity test on reserved ML roles
