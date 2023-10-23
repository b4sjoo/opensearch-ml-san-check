#!/bin/sh
import os
import argparse
import json
import time
import glob
import datetime
import warnings
import requests
from requests.auth import HTTPBasicAuth
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

parser = argparse.ArgumentParser()
parser.add_argument(
    "host", type=str, help="The host address with explicitly specifying the protocol (http or https)")
parser.add_argument("port", type=str, help="The port number")
parser.add_argument("--auth", nargs=2, help="The authentication for logging to your OpenSearch cluster, "
                    "with your first parameter being account and second being your password, split by space. "
                    "Please note that if --auth is not specified, the security test won't be performed.")
parser.add_argument("--working_directory", "-wd", nargs='?', const=os.getcwd(),
                    help="The directory storing sample data. Data loading process will be activated if specified.")
parser.add_argument("--model_group_name", "-N", nargs='?', default='auto_sancheck_embedding_group', help="A unique model group name.")
parser.add_argument("--ml_node_only", "-ML", action='store_true',
                    help="Whether the ml commons plugin can be run on all nodes or can only be run on ml nodes."
                         "If not specified, the ml commons plugin can be run on all nodes.")
parser.add_argument("--memory_cb_activate", "-CB", action='store_true',
                    help="Whether to deactivate the memory circuit breaker or not. If not specified, "
                         "the circuit breaker is deactivated")
parser.add_argument("--pretrained_model_only", "-PM", action='store_true',
                    help="Whether to enable registering model via url. If not specified, "
                         "we can only use opensearch pre-trained models")

args = parser.parse_args()
URL = f"{args.host}:{args.port}/"
if args.auth is not None:
    AUTH = HTTPBasicAuth(*args.auth)
else:
    AUTH = None

WORKING_DIRECTORY = args.working_directory

MODEL_GROUP_NAME = args.model_group_name + "_" + str(hash(time.time()))[-4:]
HEADERS = {'Content-type': 'application/json'}
ML_NODE_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.only_run_on_ml_node": False}})
MEMORY_CB_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.native_memory_threshold": 100}})
TRUSTED_URL_MODEL_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.allow_registering_model_via_url": True}})
TRUSTED_LOCAL_MODEL_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.allow_registering_model_via_local_file": True}})

san_check_id_dict = dict()


def config_preparation(ml_node_config=False, memory_cb_config=False, pretrained_model_config=False):
    print(f"{datetime.datetime.now()} Pre-check the connection to the test cluster")
    assert requests.get(URL+f"_cluster/settings", auth=AUTH, verify=False).ok is True, \
        "Failed to get cluster settings, please check your network or OpenSearch configuration."
    if not ml_node_config:
        print(f"{datetime.datetime.now()} Updating ml_node settings")
        response = requests.put(URL+f"_cluster/settings", data=ML_NODE_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override ml_node settings, return {response.json()}, please check your network or OpenSearch configuration."
    if not memory_cb_config:
        print(f"{datetime.datetime.now()} Updating memory circuit breaker settings")
        response = requests.put(URL+f"_cluster/settings", data=MEMORY_CB_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override memory circuit breaker settings, return {response.json()}, please check your network or OpenSearch configuration."
    if not pretrained_model_config:
        print(f"{datetime.datetime.now()} Updating registering model via url settings")
        response = requests.put(URL+f"_cluster/settings", data=TRUSTED_URL_MODEL_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override registering model via url settings, return {response.json()}, please check your network or OpenSearch configuration."    
        print(f"{datetime.datetime.now()} Updating registering model via local settings")
        response = requests.put(URL+f"_cluster/settings", data=TRUSTED_LOCAL_MODEL_OVERRIDE,
                                headers=HEADERS, auth=AUTH, verify=False)
        assert response.ok is True, \
            f"Failed to override registering model via local settings, return {response.json()}, please check your network or OpenSearch configuration."    


def data_preparation(data_path):
    if data_path[-1] != '/':
        data_path += '/'
    with open(data_path + 'data_indices.json') as indices_file:
        data_indices = json.load(indices_file)
    index_repr_dict = {"semantic_demostore": "semantic search sample data"}
    for (index_key, index_representation) in index_repr_dict.items():
        print(f"{datetime.datetime.now()} Creating {index_representation} index.")
        response = requests.put(URL + index_key, data=json.dumps(data_indices[index_key]),
                                headers=HEADERS, auth=AUTH, verify=False)
        try:
            assert response.ok is True
        except AssertionError:
            print(f"Creating {index_representation} index failed.")
            continue

        print(f"{datetime.datetime.now()} Begin ingesting {index_representation}.")
        with open(data_path + f'{index_key}.json', 'r') as f:
            lines = f.readlines()
            request_body = ''
            i = 0
            while i < len(lines):
                request_body += ''.join([json.dumps(json.loads(x)) +
                                        '\n' for x in lines[i:i+2]])
                i += 2
                # Optimal bulk load is 5000 data
                if i % 10000 == 0:
                    response = requests.post(URL + f"_bulk", data=request_body,
                                             headers=HEADERS, auth=AUTH, verify=False)
                    assert response.ok is True, f"Bulk load {index_representation} failed."
                    print(
                        f"{datetime.datetime.now()} Everything normal, wait for 3 sec.")
                    time.sleep(3)
                    request_body = ''
            # Bulk load final lines
            if request_body != '':
                response = requests.post(URL + f"_bulk", data=request_body,
                                         headers=HEADERS, auth=AUTH, verify=False)
                assert response.ok is True, f"Bulk load {index_representation} failed."
            print(
                f"{datetime.datetime.now()} Bulk load {index_representation} end successfully.")


def model_register(request_body, model_name, model_format, case="pretrained"):
    response = requests.post(URL+f"_plugins/_ml/models/_register", data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "CREATED", \
            f"{case.capitalize()} {model_name} {model_format} model task not created, returned {response.json()}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'{case.capitalize()} {model_name} {model_format} model task not created '
              f'with status code {response.status_code}')
    san_check_id_dict.update(
        {case: {"task_id": response.json()["task_id"]}})
    
    time.sleep(10)
    response = requests.get(URL+f"_plugins/_ml/tasks/{san_check_id_dict[case]['task_id']}",
                            auth=AUTH, verify=False)
    try:
        assert response.json()["state"] == "COMPLETED", \
            f"{case.capitalize()} {model_name} {model_format} model task not completed, returned {response.json()}, please check."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{case.capitalize()} {model_name} {model_format} model task not completed, "
              f"with status code {response.status_code}")
    san_check_id_dict[case].update({"model_id": response.json()["model_id"]})
      
    response = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    try:
        assert response.json()["model_state"] == "REGISTERED", \
            f"Failed to register {case} {model_name} {model_format} model, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        f"Failed to register {case} {model_name} {model_format} model, returned {response.text}, please check your network or OpenSearch configuration."

    response = requests.get(URL+f"_plugins/_ml/profile/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    assert response.json() == {}, \
        f"Failed to profile undeployed {case} {model_name} {model_format} model, returned {response.text}, please check your network or OpenSearch configuration."
    print(f"{datetime.datetime.now()} Successfully registered {case} {model_name} {model_format} model.")


def model_deployer(model_name, model_format, case="pretrained"):
    response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}/_deploy", auth=AUTH, verify=False)
    time.sleep(3)
    response = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    try:
        assert response.json()["model_state"] == "DEPLOYED", \
            f"Failed to deploy {case} {model_name} {model_format} model, returned model_state {response.json()['model_state']}, please check your network or OpenSearch configuration."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{case.capitalize()} {model_name} {model_format} model deployment failed with response {response.text}, please check.")
    
    response = requests.get(URL+f"_plugins/_ml/profile/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    try:
        assert response.json()["nodes"], \
            f"Failed to profile deployed {case} {model_name} {model_format} model, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"case.capitalize() {model_name} {model_format} model deployment failed with response {response.text}, please check.")
    print(f"{datetime.datetime.now()} Successfully deployed {case} {model_name} {model_format} model.")


def model_undeployer(model_name, model_format, case="pretrained"):
    response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}/_undeploy", auth=AUTH, verify=False)
    time.sleep(1)
    response = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    try:
        assert response.json()["model_state"] == "UNDEPLOYED", \
            f"Failed to undeploy {case} {model_name} {model_format} model, returned model_state {response.json()['model_state']}, please check your network or OpenSearch configuration."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{case.capitalize()} {model_name} {model_format} model undeployment failed with response {response.text}, please check.")
    
    response = requests.get(URL+f"_plugins/_ml/profile/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    assert response.json() == {}, \
        f"Failed to profile undeployed {case} {model_name} {model_format} model, returned {response.text}, please check your network or OpenSearch configuration."
    print(f"{datetime.datetime.now()} Successfully undelpoyed {case} {model_name} {model_format} model.")

    response = requests.delete(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False)
    assert response.ok, \
            f"Failed to delete undeployed {case} {model_name} {model_format} model, returned {response.text}, please check your network or OpenSearch configuration."
    print(f"{datetime.datetime.now()} Successfully deleted {case} {model_name} {model_format} model.")


def model_prediction_test(model_name, model_format, case="pretrained"):
    DATA = json.dumps({"text_docs":[ "today is sunny"], "return_number": True, "target_response": ["sentence_embedding"]
})
    dimension = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}", auth=AUTH, verify=False).json()["model_config"]["embedding_dimension"]
    
    print(f"{datetime.datetime.now()} Start to test old predict API on {case} {model_name} {model_format} model.")
    response = requests.post(URL+f"_plugins/_ml/_predict/text_embedding/{san_check_id_dict[case]['model_id']}", data=DATA, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["inference_results"][0]["output"][0]["shape"][0] == \
            len(response.json()["inference_results"][0]["output"][0]["data"]), \
            f"Failed to test old predict API on {case} {model_name} {model_format} model, returned {response.json()}, please check your network or OpenSearch configuration."
        assert response.json()["inference_results"][0]["output"][0]["shape"][0] == dimension, \
            f"Failed to test old predict API on {case} {model_name} {model_format} model, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{case.capitalize()} {model_name} {model_format} model prediction failed with response {response.text}, please check.")
    print(f"{datetime.datetime.now()} Successfully tested old predict API on {case} {model_name} {model_format} model."

    f"{datetime.datetime.now()} Start to test new predict API on {case} {model_name} {model_format} model.")
    response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict[case]['model_id']}/_predict", data=DATA, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["inference_results"][0]["output"][0]["shape"][0] == \
            len(response.json()["inference_results"][0]["output"][0]["data"]), \
            f"Failed to test new predict API on {case} {model_name} {model_format} model, returned {response.json()}, please check your network or OpenSearch configuration."
        assert response.json()["inference_results"][0]["output"][0]["shape"][0] == dimension, \
            f"Failed to test new predict API on {case} {model_name} {model_format} model, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        print(f"{datetime.datetime.now()} "
              f"{case.capitalize()} {model_name} {model_format} model prediction failed with response {response.text}, please check.")
    print(f"{datetime.datetime.now()} Successfully tested new predict API on {case} {model_name} {model_format} model.")


def model_retriver(model_name, url, data_path, chunksize=10000000):
    with requests.get(url, stream=True, verify=False) as response:
        if response.ok:
            print(f"{datetime.datetime.now()} Model retrival success. Now start to write model chunks.")
            for i, chunk in enumerate(response.iter_content(chunk_size=chunksize)):
                with open(f"{data_path}{model_name}_{i}", 'wb') as chunk_file:
                    chunk_file.write(chunk)
                print(f"{datetime.datetime.now()} Model chunk {i} writing end successfully.")
            print(f"{datetime.datetime.now()} Model chunks writing end successfully.")
            return i + 1
        else:
            print(f"{datetime.datetime.now()} Model retrival failed.")
            return response.text
        

def embedding_model_registration(model_name, model_format, url=None, model_config_path=None):
    print(f"{datetime.datetime.now()} Start to register pretrained {model_name} {model_format} model.")
    PRETRAINED_MODEL_REQUEST_BODY = json.dumps({
            "name": f"huggingface/sentence-transformers/{model_name}",
            "version": "1.0.1",
            "model_format": f"{model_format}",
            "model_group_id": f"{san_check_id_dict['model_group_id']}"
        })
    
    model_register(PRETRAINED_MODEL_REQUEST_BODY, model_name, model_format, "pretrained")
    model_deployer(model_name, model_format, "pretrained")
    model_prediction_test(model_name, model_format, case="pretrained")
    model_undeployer(model_name, model_format, "pretrained")

    if url:
        print(f"{datetime.datetime.now()} Start to register {model_name} {model_format} model from url.")
        response = requests.get(f"{url}{model_format}/config.json", verify=False)
        assert response.ok, f"{datetime.datetime.now()} Model config retrival failed, returned {response.text}."
        print(f"{datetime.datetime.now()} Model config retrival success. Now start to build url model request body.")
        model_registration = response.json()
        model_registration.pop("created_time")
        model_registration.pop("model_content_size_in_bytes")
        model_registration["url"] = f"{url}{model_format}/sentence-transformers_{model_name}-1.0.1-{model_format}.zip"
        print(f"{datetime.datetime.now()} Successfully build url model request body.")

        model_register(json.dumps(model_registration), model_name, model_format, "url")
        model_deployer(model_name, model_format, "url")
        model_prediction_test(model_name, model_format, case="url")
        model_undeployer(model_name, model_format, "url")
    
    if model_config_path:
        print(f"{datetime.datetime.now()} Start to register {model_name} {model_format} model from local.")
        if model_config_path[-1] != '/':
            model_config_path += '/'
        with open(f"{model_config_path}{model_name}-{model_format}-config.json") as f:
            model_registration = json.load(f)
        model_registration["model_group_id"] = san_check_id_dict["model_group_id"]
        total_chunks = model_retriver("all-MiniLM-L12-v2", "https://artifacts.opensearch.org/models/ml-models/huggingface/sentence-transformers/all-MiniLM-L12-v2/1.0.1/torch_script/sentence-transformers_all-MiniLM-L12-v2-1.0.1-torch_script.zip", model_config_path, chunksize=10000000)
        model_registration["total_chunks"] = total_chunks
        print(f"{datetime.datetime.now()} Successfully build local model meta request body.")
        
        response = requests.post(URL+f"_plugins/_ml/models/meta", data=json.dumps(model_registration), headers=HEADERS, auth=AUTH, verify=False)
        try:
            san_check_id_dict.update({"local": {"model_id": response.json()['model_id']}})
        except KeyError:
            print(f"{datetime.datetime.now()} "
                f"Local {model_name} {model_format} model meta registration failed with response {response.text}, please check.")
        print(f"{datetime.datetime.now()} Successfully registered local {model_name} {model_format} model meta.")
        
        san_check_id_dict["local"].update({"total_chunks": response.json()['model_id']})
        print(san_check_id_dict)
        for i in range(model_registration["total_chunks"]):
            with open(f"{model_config_path}{model_name}_{i}", 'rb') as chunk_file:
                response = requests.post(URL+f"_plugins/_ml/models/{san_check_id_dict['local']['model_id']}/chunk/{i}", data=chunk_file, auth=AUTH, verify=False, headers=HEADERS)
                assert response.ok, \
                    f"{datetime.datetime.now()} Local {model_name} {model_format} model chunk {i} registration failed with response {response.text}, please check."
            print(f"{datetime.datetime.now()} Successfully uploaded local {model_name} {model_format} model chunk {i}.")
        print(f"{datetime.datetime.now()} Successfully uploaded local {model_name} {model_format} model.")
        model_deployer(model_name, model_format, "local")
        model_prediction_test(model_name, model_format, case="local")
        model_undeployer(model_name, model_format, "local")
        for f in glob.glob(model_config_path + model_name + "_*"):
            os.remove(f)


def main():
    config_preparation(args.ml_node_only, args.memory_cb_activate, args.pretrained_model_only)
    print(f"{datetime.datetime.now()} Remote inference sanity check started.")
    model_group_registration = json.dumps({"name": MODEL_GROUP_NAME,
                                           "description": "This model group is created automatically through the python script used for the text emdding models' sanity test"})
    response = requests.post(URL+f"_plugins/_ml/model_groups/_register", data=model_group_registration, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "CREATED", \
            f"Failed to register model group, returned {response.json()}, please check your network or OpenSearch configuration."
    except KeyError:
        assert response.ok, \
        f"Failed to register model group, returned {response.json()}, please check your network or OpenSearch configuration."
    san_check_id_dict["model_group_id"] = response.json()['model_group_id']
    
#    if WORKING_DIRECTORY:
#        data_preparation(WORKING_DIRECTORY)
    print(f"{datetime.datetime.now()} Sanity check started.")
    
    embedding_model_registration("all-MiniLM-L12-v2", "torch_script", "https://artifacts.opensearch.org/models/ml-models/huggingface/sentence-transformers/all-MiniLM-L12-v2/1.0.1/", WORKING_DIRECTORY)

    response = requests.delete(URL+f"_plugins/_ml/model_groups/{san_check_id_dict['model_group_id']}", auth=AUTH, verify=False)
    assert response.ok, \
            f"Failed to delete model group, returned {response.text}, please check your network or OpenSearch configuration."
    print(f"{datetime.datetime.now()} Successfully deleted model group.")


if __name__ == '__main__':
    main()