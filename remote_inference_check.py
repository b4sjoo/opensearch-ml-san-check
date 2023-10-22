#!/bin/sh
import os
import argparse
import json
import time
import datetime
import warnings
import requests
import random
import configparser
from requests.auth import HTTPBasicAuth
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

parser = argparse.ArgumentParser()
parser.add_argument(
    "host", type=str, help="The host address with explicitly specifying the protocol (http or https)")
parser.add_argument("port", type=str, help="The port number")
parser.add_argument("--auth", nargs=2, help="The authentication for logging to your OpenSearch cluster, "
                    "with your first parameter being account and second being your password, split by space. "
                    "Please note that if --auth is not specified, the security test won't be performed.")
parser.add_argument("--connector_config_directory", "-cd", type=str, help="The config file storing API secrets of your remote inference endpoints.")
parser.add_argument("--model_group_name", "-N", nargs='?', default='auto_sancheck_connector_group', help="A unique model group name.")
parser.add_argument("--ml_node_only", "-ML", action='store_true',
                    help="Whether the ml commons plugin can be run on all nodes or can only be run on ml nodes."
                         "If not specified, the ml commons plugin can be run on all nodes.")
parser.add_argument("--memory_cb_activate", "-CB", action='store_true',
                    help="Whether to deactivate the memory circuit breaker or not. If not specified, "
                         "the circuit breaker is deactivated")
parser.add_argument("--disable_connectors", "-D", action='store_true',
                    help="Whether to enable the remote inference or not. If specified, "
                         "we will forbid all the connectors endpoint.")

args = parser.parse_args()
URL = f"{args.host}:{args.port}/"

if args.auth is not None:
    AUTH = HTTPBasicAuth(*args.auth)
else:
    AUTH = None

CONFIG_DIRECTORY = args.connector_config_directory
if CONFIG_DIRECTORY[-1] != '/':
    CONFIG_DIRECTORY += '/'

secret_config = configparser.ConfigParser()
with open(CONFIG_DIRECTORY + 'secret_config.ini') as f:
    secret_config.read_file(f)
with open(CONFIG_DIRECTORY + 'connector_config.json') as f:
    connector_config = json.load(f)

MODEL_GROUP_NAME = args.model_group_name + "_" + str(hash(time.time()))[-4:]
HEADERS = {'Content-type': 'application/json'}
ML_NODE_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.only_run_on_ml_node": False}})
MEMORY_CB_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.native_memory_threshold": 100}})

san_check_id_dict = dict()

def config_preparation(ml_node_config=False, memory_cb_config=False, disable_connectors_config=False):
    print(f"{datetime.datetime.now()} Pre-check the connection to the test cluster")
    assert requests.get(URL+f"_cluster/settings", auth=AUTH, verify=False).ok is True, \
        "Failed to get cluster settings, please check your network or OpenSearch configuration."
    if not ml_node_config:
        print(f"{datetime.datetime.now()} Updating ml_node settings")
        response = requests.put(URL+f"_cluster/settings", data=ML_NODE_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override ml_node settings, returned {response.json()}, please check your network or OpenSearch configuration."
    if not memory_cb_config:
        print(f"{datetime.datetime.now()} Updating memory circuit breaker settings")
        response = requests.put(URL+f"_cluster/settings", data=MEMORY_CB_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override memory circuit breaker settings, returned {response.json()}, please check your network or OpenSearch configuration."
    if not disable_connectors_config:
        TRUSTED_CONNECTORS_OVERRIDE = json.dumps({"persistent": {
            "plugins.ml_commons.trusted_connector_endpoints_regex": [
                "^https://runtime\\.sagemaker\\..*\\.amazonaws\\.com/.*$",
                "^https://api\\.openai\\.com/.*$",
                "^https://api\\.cohere\\.ai/.*$",
                "^https://bedrock\\..*\\.amazonaws.com/.*$"]}})
        print(f"{datetime.datetime.now()} Updating trusted connector endpoint settings")
        response = requests.put(URL+f"_cluster/settings", data=TRUSTED_CONNECTORS_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override trusted connector endpoint settings, return {response.json()}, please check your network or OpenSearch configuration."
    else:
        TRUSTED_CONNECTORS_OVERRIDE = json.dumps({"persistent": {
            "plugins.ml_commons.trusted_connector_endpoints_regex": []}})
        print(f"{datetime.datetime.now()} Disabling trusted connector endpoint settings")
        response = requests.put(URL+f"_cluster/settings", data=TRUSTED_CONNECTORS_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override trusted connector endpoint settings, returned {response.json()}, please check your network or OpenSearch configuration."
        

def create_connector(model_type):
    print(f"{datetime.datetime.now()} Creating {model_type} connector")
    connector_registration = connector_config[model_type + "_connector"]

    if model_type == "openai":
        connector_registration["credential"] = {"openAI_key": f"{secret_config['THIRD_PARTY_MODEL_API_KEY']['OPENAI_KEY']}"}
    elif model_type == "cohere":
        connector_registration["credential"] = {"cohere_key": f"{secret_config['THIRD_PARTY_MODEL_API_KEY']['COHERE_KEY']}"}
    elif model_type == "bedrock":
        connector_registration["credential"] = {"access_key": f"{secret_config['BEDROCK_SECRET_CONFIG']['AWS_ACCESS_KEY_ID']}",
                                                "secret_key": f"{secret_config['BEDROCK_SECRET_CONFIG']['AWS_SECRET_ACCESS_KEY']}",
                                                "session_token": f"{secret_config['BEDROCK_SECRET_CONFIG']['AWS_SESSION_TOKEN']}"}
        connector_registration["parameters"] = {"region": "us-east-1", "service_name": "bedrock"}
    elif model_type == "sagemaker":
        connector_registration["credential"] = {"access_key": f"{secret_config['SAGEMAKER_SECRET_CONFIG']['AWS_ACCESS_KEY_ID']}",
                                                "secret_key": f"{secret_config['SAGEMAKER_SECRET_CONFIG']['AWS_SECRET_ACCESS_KEY']}",
                                                "session_token": f"{secret_config['SAGEMAKER_SECRET_CONFIG']['AWS_SESSION_TOKEN']}"}
        connector_registration["parameters"] = {"region": "us-east-1", "service_name": "sagemaker"}
    else:
        raise ValueError("Invalid model type, please check your config file.")
    response = requests.post(URL+f"_plugins/_ml/connectors/_create", data=json.dumps(connector_registration), headers=HEADERS, auth=AUTH, verify=False)
    try:
        san_check_id_dict[model_type] = {"connector_id": response.json()["connector_id"]}
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f"{model_type} standalone connector creation failed with response {response.text}, please check.")
    
    model_registration = connector_config[f"{model_type}_model"]
    model_registration["model_group_id"] = san_check_id_dict["model_group_id"]
    model_registration["connector_id"] = san_check_id_dict[model_type]["connector_id"]
    
    print(f"{datetime.datetime.now()} Creating {model_type} connector completed.")

    return connector_registration, model_registration


def deploy_remote_model(model_type, connector_type, connector_registration, model_registration):
    print(f"{datetime.datetime.now()} Deploying {model_type} {connector_type} connector.")

    if connector_type == "internal":
        model_registration.pop("connector_id")
        model_registration["connector"] = connector_registration
    elif connector_type == "standalone":
        pass
    else:
        raise ValueError("Invalid connector type, please choose between \"standalone\" and \"internal\".")

    response = requests.post(URL+f"_plugins/_ml/models/_register", data=json.dumps(model_registration), headers=HEADERS, auth=AUTH, verify=False)
    try:
        san_check_id_dict[model_type].update({f"{connector_type}_connector_model": response.json()['model_id']})
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{model_type} {connector_type} connector model registration failed with response {response.text}, please check.")
    time.sleep(1)

    response = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict[model_type][f'{connector_type}_connector_model']}", auth=AUTH, verify=False)
    try:
        assert response.json()["model_state"] == "REGISTERED", \
            f"Failed to register {connector_type} connector model, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        f"Failed to register {connector_type} connector model, returned {response.text}, please check your network or OpenSearch configuration."
    response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict[model_type][f'{connector_type}_connector_model']}/_deploy", auth=AUTH, verify=False)
    time.sleep(3)
    
    response = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict[model_type][f'{connector_type}_connector_model']}", auth=AUTH, verify=False)
    try:
        assert response.json()["model_state"] == "DEPLOYED", \
            f"Failed to deploy {connector_type} connector model, returned model_state {response.json()['model_state']}, please check your network or OpenSearch configuration."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{model_type} {connector_type} connector model deployment failed with response {response.text}, please check.")
        
    print(f"{datetime.datetime.now()} Successfully deployed {model_type} {connector_type} connector.")


def repeat_connector_test(model_type, test_type="MIX", times=3):
    response_list = list()

    if test_type == "MIX" or test_type == "STANDALONE":
        print(f"{datetime.datetime.now()} Testing {model_type} standalone connector.")
        for i in range(times):
            print(f"{datetime.datetime.now()} This is the {i+1} time to test {model_type} standalone connector.")
            if model_type == "openai":
                DATA = json.dumps({"parameters": {"prompt": "Say this is a test"}})
            elif model_type == "cohere":
                DATA = json.dumps({"parameters": {"prompt": ["Say this is a test"]}})
            elif model_type == "bedrock":
                DATA = json.dumps({"parameters": {"inputText": "Say this is a test"}})
            elif model_type == "sagemaker":
                DATA = json.dumps({"parameters": {"inputs": "Hello world"}})
            else:
                raise ValueError("Invalid model type, please check your config file.")
            response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict[model_type]['standalone_connector_model']}/_predict", data=DATA, headers=HEADERS, auth=AUTH, verify=False)
            response_list.append(response)
            time.sleep(random.uniform(0, 1))
        print(f"{datetime.datetime.now()} {model_type} standalone connector prediction result: {response.json()}")
    if test_type == "MIX" or test_type == "STANDALONE":
        print(f"{datetime.datetime.now()} Testing {model_type} internal connector.")
        for i in range(times):
            print(f"{datetime.datetime.now()} This is the {i+1} time to test {model_type} internal connector.")
            if model_type == "openai":
                DATA = json.dumps({"parameters": {"prompt": "Say this is a test"}})
            elif model_type == "cohere":
                DATA = json.dumps({"parameters": {"prompt": ["Say this is a test"]}})
            elif model_type == "bedrock":
                DATA = json.dumps({"parameters": {"inputText": "Say this is a test"}})
            elif model_type == "sagemaker":
                DATA = json.dumps({"parameters": {"inputs": "Hello world"}})
            else:
                raise ValueError("Invalid model type, please check your config file.")
            response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict[model_type]['internal_connector_model']}/_predict", data=DATA, headers=HEADERS, auth=AUTH, verify=False)
            response_list.append(response)
            time.sleep(random.uniform(0, 1))
        print(f"{datetime.datetime.now()} {model_type} internal connector prediction result: {response.json()}")

def main():
    # Initialization
    config_preparation(args.ml_node_only, args.memory_cb_activate, args.disable_connectors)
    print(f"{datetime.datetime.now()} Remote inference sanity check started.")
    model_group_registration = json.dumps({"name": MODEL_GROUP_NAME,
                                           "description": "This model group is created automatically through the python script used for the remote inference sanity test"})
    response = requests.post(URL+f"_plugins/_ml/model_groups/_register", data=model_group_registration, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "CREATED", \
            f"Failed to register model group, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        assert response.ok, \
        f"Failed to register model group, returned {response.json()}, please check your network or OpenSearch configuration."
    san_check_id_dict["model_group_id"] = response.json()['model_group_id']

    connector_registration, model_registration = create_connector("openai")
    deploy_remote_model("openai", "standalone", connector_registration, model_registration)
    deploy_remote_model("openai", "internal", connector_registration, model_registration)

    connector_registration, model_registration = create_connector("cohere")
    deploy_remote_model("cohere", "standalone", connector_registration, model_registration)
    deploy_remote_model("cohere", "internal", connector_registration, model_registration)

    connector_registration, model_registration = create_connector("bedrock")
    deploy_remote_model("bedrock", "standalone", connector_registration, model_registration)
    deploy_remote_model("bedrock", "internal", connector_registration, model_registration)

    connector_registration, model_registration = create_connector("sagemaker")
    deploy_remote_model("sagemaker", "standalone", connector_registration, model_registration)
    deploy_remote_model("sagemaker", "internal", connector_registration, model_registration)

    print(f"{datetime.datetime.now()} Sanity test dict: {san_check_id_dict}.")
    repeat_connector_test("openai", "MIX", 1)
    repeat_connector_test("cohere", "MIX", 1)
    repeat_connector_test("bedrock", "MIX", 1)
    repeat_connector_test("sagemaker", "MIX", 1)
if __name__ == '__main__':
    main()
