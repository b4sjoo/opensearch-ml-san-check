#!/bin/sh
import os
import argparse
import json
import time
import datetime
import warnings
import requests
from requests.auth import HTTPBasicAuth
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

parser = argparse.ArgumentParser()
parser.add_argument(
    "host", type=str, help="The host address with explicitly specifying the protocol (http or https)")
parser.add_argument("port", type=str, help="The port number")
parser.add_argument("--working_directory", "-wd", nargs='?', const=os.getcwd(),
                    help="The directory storing sample data. Data loading process will be activated if specified.")
parser.add_argument("--auth", nargs=2, help="The authentication for logging to your OpenSearch cluster, "
                    "with your first parameter being account and second being your password, split by space. "
                    "Please note that if --auth is not specified, the security test won't be performed.")
parser.add_argument("--ml_node_only", "-ML", action='store_true',
                    help="Whether the ml commons plugin can be run on all nodes or can only be run on ml nodes."
                         "If not specified, the ml commons plugin can be run on all nodes.")
parser.add_argument("--memory_cb_activate", "-CB", action='store_true',
                    help="Whether to deactivate the memory circuit breaker or not. If not specified, "
                         "the circuit breaker is deactivated")

args = parser.parse_args()
URL = f"{args.host}:{args.port}/"
if args.auth is not None:
    AUTH = HTTPBasicAuth(*args.auth)
else:
    AUTH = None
WORKING_DIRECTORY = args.working_directory

HEADERS = {'Content-type': 'application/json'}
ML_NODE_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.only_run_on_ml_node": False}})
MEMORY_CB_OVERRIDE = json.dumps(
    {"persistent": {"plugins.ml_commons.native_memory_threshold": 100}})

KMEANS_DIRECT_INPUT_DATA = json.dumps({
    "parameters": {"centroids": 2, "iterations": 1, "distance_type": "EUCLIDEAN"},
    "input_data": {"column_metas": [
        {"name": "k1", "column_type": "DOUBLE"},
        {"name": "k2", "column_type": "DOUBLE"}],
        "rows": [
        {"values": [
            {"column_type": "DOUBLE", "value": 1.00},
            {"column_type": "DOUBLE", "value": 2.00}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 1.00},
            {"column_type": "DOUBLE", "value": 4.00}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 1.00},
            {"column_type": "DOUBLE", "value": 0.00}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 10.00},
            {"column_type": "DOUBLE", "value": 2.00}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 10.00},
            {"column_type": "DOUBLE", "value": 4.00}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 10.00},
            {"column_type": "DOUBLE", "value": 0.00}]}]}})

LINEAR_DIRECT_INPUT_DATA = json.dumps({
    "parameters": {"target": "price"},
    "input_data": {"column_metas": [
        {"name": "A", "column_type": "DOUBLE"},
        {"name": "B", "column_type": "DOUBLE"},
        {"name": "price", "column_type": "DOUBLE"}],
        "rows": [{
            "values": [
                {"column_type": "DOUBLE", "value": 1},
                {"column_type": "DOUBLE", "value": 1},
                {"column_type": "DOUBLE", "value": 6}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 1},
            {"column_type": "DOUBLE", "value": 2},
            {"column_type": "DOUBLE", "value": 8}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 2},
            {"column_type": "DOUBLE", "value": 2},
            {"column_type": "DOUBLE", "value": 9}]},
        {"values": [
            {"column_type": "DOUBLE", "value": 2},
            {"column_type": "DOUBLE", "value": 3},
            {"column_type": "DOUBLE", "value": 11}]}]}})

LOGI_DIRECT_INPUT_DATA = json.dumps({"parameters": {"target": "class"}, "input_data": {
    "column_metas": [
        {"name": "sepal_length_in_cm", "column_type": "DOUBLE"},
        {"name": "sepal_width_in_cm", "column_type": "DOUBLE"},
        {"name": "petal_length_in_cm", "column_type": "DOUBLE"},
        {"name": "petal_width_in_cm", "column_type": "DOUBLE"}],
    "rows": [{
        "values": [
            {"column_type": "DOUBLE", "value": 6.2}, {
                "column_type": "DOUBLE", "value": 3.4},
            {"column_type": "DOUBLE", "value": 5.4}, {"column_type": "DOUBLE", "value": 2.3}]}, {
        "values": [
            {"column_type": "DOUBLE", "value": 5.9}, {
                "column_type": "DOUBLE", "value": 3.0},
            {"column_type": "DOUBLE", "value": 5.1}, {"column_type": "DOUBLE", "value": 1.8}]}]}})


def data_preparation(data_path):
    if data_path[-1] != '/':
        data_path += '/'
    with open(data_path + 'data_indices.json') as indices_file:
        data_indices = json.load(indices_file)
    index_repr_dict = {'iris_data': "iris data", 'nyc_taxi': "nyc taxi data", 'fourclass_data': "four class data",
                       'test_data': "training data", 'predict_data': "predicting data",
                       "rca-index": "anomaly localization data"}
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


def config_preparation(ml_node_config=False, memory_cb_config=False):
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


# TODO: try to test different user roles
def security_test(id_dict):
    print(f"{datetime.datetime.now()} Testing search protected indices.")
    response = requests.get(
        URL+f".plugins-ml-task/_search", auth=AUTH, verify=False)
    try:
        assert response.json()["hits"]["total"]["value"] == 0,\
            f"Testing search protected tasks failed, " \
            f"expect task number 0, return {response.json()['hits']['total']['value']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing search protected tasks failed with status code {response.status_code}')

    response = requests.get(
        URL + f".plugins-ml-model/_search", auth=AUTH, verify=False)
    try:
        assert response.json()["hits"]["total"]["value"] == 0, \
            f"Testing search protected models failed, " \
            f"expect model number 0, return {response.json()['hits']['total']['value']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing search protected models failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing search protected indices finished.")

    print(f"{datetime.datetime.now()} Testing delete protected indices.")
    response = requests.delete(URL+f".plugins-ml-task/_doc/{id_dict['async_rcf']['task_id']}",
                               auth=AUTH, verify=False)
    try:
        assert response.json()["error"]["type"] == "security_exception",\
            f"Testing delete protected tasks failed, " \
            f"expect 'security_exception', return {response.json()['error']['root_cause']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing delete protected tasks failed failed with status code {response.status_code}')

    response = requests.delete(URL+f".plugins-ml-model/_doc/{id_dict['async_rcf']['model_id']}",
                               auth=AUTH, verify=False)
    try:
        assert response.json()["error"]["type"] == "security_exception",\
            f"Testing delete protected models failed, " \
            f"expect 'security_exception', return {response.json()['error']['root_cause']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing delete protected models failed failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing delete protected indices finished.")


def main():
    config_preparation(args.ml_node_only, args.memory_cb_activate)
    if WORKING_DIRECTORY is not None:
        data_preparation(WORKING_DIRECTORY)
    print(f"{datetime.datetime.now()} Sanity check started.")
    san_check_id_dict = dict()

    # Training then predicting logistic regression with iris data in async way
    print(f"{datetime.datetime.now()} Training logistic regression with iris data in async way.")
    request_body = json.dumps({"parameters": {"target": "class"},
                               "input_query": {"query": {"match_all": {}},
                                               "_source": ["sepal_length_in_cm", "sepal_width_in_cm",
                                                           "petal_length_in_cm", "petal_width_in_cm", "class"],
                                               "size": 200},
                               "input_index": ["iris_data"]})
    response = requests.post(URL+f"_plugins/_ml/_train/logistic_regression?async=true", data=request_body,
                             headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "CREATED", \
            "Training logistic regression with iris data in async way not created, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training logistic regression with iris data in async way not created '
              f'with status code {response.status_code}')
    san_check_id_dict.update(
        {"async_logi": {"task_id": response.json()["task_id"]}})
    time.sleep(3)
    response = requests.get(URL+f"_plugins/_ml/tasks/{san_check_id_dict['async_logi']['task_id']}",
                            auth=AUTH, verify=False)
    try:
        assert response.json()["state"] == "COMPLETED", \
            "Training logistic regression with iris data in async way not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training logistic regression with iris data in async way not completed '
              f'with status code {response.status_code}')
    san_check_id_dict["async_logi"].update(
        {"model_id": response.json()["model_id"]})
    print(f"{datetime.datetime.now()} Training logistic regression with iris data in async way finished.")

    print(f"{datetime.datetime.now()} Predicting logistic regression with iris data.")
    request_body = LOGI_DIRECT_INPUT_DATA
    response = requests.post(URL+f"_plugins/_ml/_predict/logistic_regression/{san_check_id_dict['async_logi']['model_id']}",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Predicting logistic regression with iris data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Predicting logistic regression with iris data not completed with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) == 2, \
        "The number of the prediction result is different from iris data, please check."
    print(f"{datetime.datetime.now()} Predicting logistic regression with iris data finished.")

    # Training then Predicting Kmeans with iris data in sync way
    request_body = json.dumps({"parameters": {"centroids": 3, "iterations": 10, "distance_type": "COSINE"},
                               "input_query": {"_source": ["petal_length_in_cm", "petal_width_in_cm"], "size": 10000},
                               "input_index": ["iris_data"]})
    print(f"{datetime.datetime.now()} Training Kmeans with iris data in sync way.")
    response = requests.post(URL+f"_plugins/_ml/_train/kmeans", data=request_body,
                             headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Training Kmeans with iris data in sync way not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training Kmeans with iris data in sync way not completed with status code {response.status_code}')
    san_check_id_dict.update(
        {"sync_kmeans": {"model_id": response.json()["model_id"]}})
    print(f"{datetime.datetime.now()} Training Kmeans with iris data in sync way finished.")

    print(f"{datetime.datetime.now()} Predicting Kmeans with iris data.")
    request_body = json.dumps({"input_query": {"_source": ["petal_length_in_cm", "petal_width_in_cm"], "size": 10000},
                               "input_index": ["iris_data"]})
    response = requests.post(URL+f"_plugins/_ml/_predict/kmeans/{san_check_id_dict['sync_kmeans']['model_id']}",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Predicting Kmeans with iris data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Predicting Kmeans with iris data not completed with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) == 150, \
        "The number of the prediction result is different from iris data, please check."
    print(f"{datetime.datetime.now()} Predicting Kmeans with iris data finished.")

    # Train and predict Kmeans with iris data
    print(f"{datetime.datetime.now()} Training and predicting Kmeans with iris data.")
    request_body = json.dumps({"input_query": {"_source": ["petal_length_in_cm", "petal_width_in_cm"], "size": 10000},
                               "input_index": ["iris_data"]})
    response = requests.post(URL+f"_plugins/_ml/_train_predict/kmeans",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Training and predicting Kmeans with iris data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training and predicting Kmeans with iris data not completed with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) == 150, \
        "The number of the training and predicting result is different from iris data, please check."
    print(f"{datetime.datetime.now()} Training and predicting Kmeans with iris data finished.")

    # Training then Predicting Kmeans with direct input data in sync way
    print(f"{datetime.datetime.now()} Training Kmeans with direct input data.")
    request_body = KMEANS_DIRECT_INPUT_DATA
    response = requests.post(URL+f"_plugins/_ml/_train/kmeans", data=request_body,
                             headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Training Kmeans with direct input data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training Kmeans with direct input data not completed with status code {response.status_code}')
    san_check_id_dict.update(
        {"direct_input_kmeans": {"model_id": response.json()["model_id"]}})
    print(f"{datetime.datetime.now()} Training Kmeans with direct input data finished.")

    print(f"{datetime.datetime.now()} Predicting Kmeans with direct input data.")
    request_body = KMEANS_DIRECT_INPUT_DATA
    response = requests.post(URL+f"_plugins/_ml/_predict/kmeans/"
                                 f"{san_check_id_dict['direct_input_kmeans']['model_id']}",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Predicting Kmeans with direct input data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Predicting Kmeans with direct input data not completed with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) == 6, \
        "The number of the prediction result is different from direct input data, please check."
    print(f"{datetime.datetime.now()} Predicting Kmeans with direct input data finished.")

    # Train and predict Kmeans with direct input data
    print(f"{datetime.datetime.now()} Training and predicting Kmeans with direct input data.")
    request_body = KMEANS_DIRECT_INPUT_DATA
    response = requests.post(URL+f"_plugins/_ml/_train_predict/kmeans",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Training and predicting Kmeans with direct input data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training and predicting Kmeans with direct input data not completed '
              f'with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) == 6, \
        "The number of the training and predicting result is different from direct input data, please check."
    print(f"{datetime.datetime.now()} Training and predicting Kmeans with direct input data finished.")

    # Training then Predicting linear regression with direct input data in sync way
    print(f"{datetime.datetime.now()} Training and predicting linear regression with direct input data.")
    request_body = LINEAR_DIRECT_INPUT_DATA
    response = requests.post(URL+f"_plugins/_ml/_train/linear_regression",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Training linear regression with direct input data in sync way not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training linear regression with direct input data in sync way not completed '
              f'with status code {response.status_code}')
    san_check_id_dict.update(
        {"sync_linear": {"model_id": response.json()["model_id"]}})
    print(f"{datetime.datetime.now()} Training linear regression with direct input data in sync way finished.")

    print(f"{datetime.datetime.now()} Predicting linear regression with direct input data.")
    request_body = json.dumps({
        "parameters": {"target": "price"},
        "input_data": {"column_metas": [{"name": "A", "column_type": "DOUBLE"},
                                        {"name": "B", "column_type": "DOUBLE"}],
                       "rows": [{"values": [{"column_type": "DOUBLE", "value": 3},
                                            {"column_type": "DOUBLE", "value": 5}]}]}})
    response = requests.post(URL+f"_plugins/_ml/_predict/linear_regression/{san_check_id_dict['sync_linear']['model_id']}",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Predicting linear regression with direct input data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Predicting linear regression with direct input data not completed '
              f'with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) == 1, \
        "The number of the prediction result is different from direct input data, please check."
    print(f"{datetime.datetime.now()} Predicting linear regression with direct input data finished.")

    # Training then Predicting RCF with nyc_taxi data(time-series data) in sync way
    print(f"{datetime.datetime.now()} Training and predicting RCF with direct input data.")
    request_body = json.dumps({"parameters": {"shingle_size": 10, "time_field": "timestamp",
                                              "date_format": "yyyy-MM-dd HH:mm:ss", "time_zone": "America/New_York"},
                               "input_query": {"query": {"bool": {"filter": [{"range": {"value": {"gte": -1}}}]}},
                                               "_source": ["timestamp", "value"], "size": 1000},
                               "input_index": ["nyc_taxi"]})
    response = requests.post(URL+f"_plugins/_ml/_train/fit_rcf",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Training RCF with nyc_taxi data in sync way not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training RCF with nyc_taxi data in sync way not completed with status code {response.status_code}')
    san_check_id_dict.update(
        {"sync_rcf": {"model_id": response.json()["model_id"]}})
    print(f"{datetime.datetime.now()} Training RCF with nyc_taxi data in sync way finished.")

    print(f"{datetime.datetime.now()} Predicting RCF with nyc_taxi data.")
    request_body = json.dumps({"parameters": {"time_field": "timestamp", "date_format": "yyyy-MM-dd HH:mm:ss",
                                              "time_zone": "America/New_York"},
                               "input_query": {"query": {"bool": {"filter": [{"range": {"value": {"gte": -1}}}]}},
                                               "_source": ["timestamp", "value"], "size": 1000},
                               "input_index": ["nyc_taxi"]})
    response = requests.post(URL+f"_plugins/_ml/_predict/fit_rcf/{san_check_id_dict['sync_rcf']['model_id']}",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Predicting RCF with nyc_taxi data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Predicting RCF with nyc_taxi data not completed '
              f'with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) <= 1000, \
        "The number of the prediction result is different from size limit, please check."
    print(f"{datetime.datetime.now()} Predicting RCF with nyc_taxi data finished.")

    # Training then predicting RCF with fourclass data(non-time-series data) in async way
    request_body = json.dumps({"parameters":
                              {"training_data_size": 200},
                               "input_query": {"query": {"match_all": {}}, "_source": ["A", "B"],
                                               "size": 10000, "sort": [{"anomaly_type": "desc"}]},
                               "input_index": ["fourclass_data"]})
    print(f"{datetime.datetime.now()} Training RCF with fourclass data in async way.")
    response = requests.post(URL+f"_plugins/_ml/_train/batch_rcf?async=true", data=request_body,
                             headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "CREATED", \
            "Training RCF with fourclass data in async way not created, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training RCF with fourclass data in async way not created with status code {response.status_code}')
    san_check_id_dict.update(
        {"async_rcf": {"task_id": response.json()["task_id"]}})
    time.sleep(5)
    response = requests.get(URL+f"_plugins/_ml/tasks/{san_check_id_dict['async_rcf']['task_id']}",
                            auth=AUTH, verify=False)
    try:
        assert response.json()["state"] == "COMPLETED", \
            "Training RCF with fourclass data in async way not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Training RCF with fourclass data in async way not completed with status code {response.status_code}')
    san_check_id_dict["async_rcf"].update(
        {"model_id": response.json()["model_id"]})
    print(f"{datetime.datetime.now()} Training RCF with fourclass data in async way finished.")

    print(f"{datetime.datetime.now()} Predicting RCF with fourclass data.")
    request_body = json.dumps({"parameters": {"anomaly_score_threshold": 0.1},
                               "input_query": {"query": {"match_all": {}}, "_source": ["A", "B"],
                                               "size": 10000, "sort": [{"anomaly_type": "desc"}]},
                               "input_index": ["fourclass_data"]})
    response = requests.post(URL+f"_plugins/_ml/_predict/batch_rcf/{san_check_id_dict['async_rcf']['model_id']}",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["status"] == "COMPLETED", \
            "Predicting RCF with fourclass data not completed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Predicting RCF with fourclass data not completed '
              f'with status code {response.status_code}')
    assert len(response.json()['prediction_result']['rows']) <= 10000, \
        "The number of the prediction result is different from size limit, please check."
    print(f"{datetime.datetime.now()} Predicting RCF with fourclass data finished.")

    # Execute model
    # Execute simple calculator model
    print(f"{datetime.datetime.now()} Testing execute API with simple calculator.")
    request_body = json.dumps(
        {"operation": "max", "input_data": [1.0, 2.0, 3.0]})
    response = requests.post(URL+f"_plugins/_ml/_execute/local_sample_calculator",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["function_name"] == "LOCAL_SAMPLE_CALCULATOR",\
            "Testing execute API with simple calculator failed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing execute API with simple calculator failed with status code {response.status_code}')
    assert response.json()['output']['result'] == 3, \
        "The sanity check for local sample calculator failed, please check."
    print(f"{datetime.datetime.now()} Testing execute API with simple calculator finished.")

    # Execute anomaly localization
    print(f"{datetime.datetime.now()} Testing execute API with anomaly localization.")
    request_body = json.dumps({"index_name": "rca-index", "attribute_field_names": ["attribute"],
                               "aggregations": [{"sum": {"sum": {"field": "value"}}}],
                               "time_field_name": "timestamp",
                               "start_time": datetime.datetime(2021, 5, 10, 0, 0).timestamp() * 1000,
                               "end_time": datetime.datetime(2021, 5, 18, 0, 0).timestamp() * 1000,
                               "min_time_interval": 86400000, "num_outputs": 10})
    response = requests.post(URL+f"_plugins/_ml/_execute/anomaly_localization",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()['output']['results'][0]['result']['buckets'][0]['overall_aggregate_value'] is not None,\
            "Testing execute API with anomaly localization failed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing execute API with anomaly localization failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing execute API with anomaly localization finished.")

    # Get model
    print(f"{datetime.datetime.now()} Testing get method with existing model_id "
          f"{san_check_id_dict['async_logi']['model_id']}.")
    response = requests.get(URL+f"_plugins/_ml/models/{san_check_id_dict['async_logi']['model_id']}",
                            auth=AUTH, verify=False)
    try:
        assert response.json()["name"] == "LOGISTIC_REGRESSION",\
            "Testing get method with existing model failed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing get method with existing model failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing get method with existing model finished.")

    print(f"{datetime.datetime.now()} Testing get method with non-existing model_id fake_model")
    response = requests.get(
        URL+f"_plugins/_ml/models/fake_model", auth=AUTH, verify=False)
    try:
        assert response.json()["error"]["type"] == "status_exception",\
            f"Testing get method with non-existing model failed, " \
            f"expect 'status_exception', return {response.json()['error']['root_cause']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing get method with non-existing model failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing get method with non-existing model finished.")

    # Search model
    print(f"{datetime.datetime.now()} Testing search API with querying all models")
    request_body = json.dumps({"query": {"match_all": {}}, "size": 1000})
    response = requests.post(URL+f"_plugins/_ml/models/_search", data=request_body, headers=HEADERS,
                             auth=AUTH, verify=False)
    try:
        assert response.json()["hits"]["total"]["value"] > 0,\
            f"Testing search API with querying all models failed, " \
            f"expect model number greater than 0, return 0, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing search API with querying all models failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing search API with querying all models finished.")

    print(f"{datetime.datetime.now()} Testing search API with querying Kmeans models")
    request_body = json.dumps(
        {"query": {"term": {"algorithm": {"value": "KMEANS"}}}})
    response = requests.post(URL+f"_plugins/_ml/models/_search", data=request_body, headers=HEADERS,
                             auth=AUTH, verify=False)
    try:
        assert response.json()["hits"]["total"]["value"] > 0,\
            f"Testing search API with querying Kmeans models failed, " \
            f"expect model number greater than 0, return 0, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing search API with querying Kmeans only models failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing search API with querying Kmeans models finished.")

    # Delete model
    print(f"{datetime.datetime.now()} Testing delete method with existing model_id "
          f"{san_check_id_dict['async_logi']['model_id']}.")
    response = requests.delete(URL+f"_plugins/_ml/models/{san_check_id_dict['async_logi']['model_id']}",
                               auth=AUTH, verify=False)
    try:
        assert response.json()["result"] == "deleted",\
            "Testing delete method with existing model failed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing delete method with existing model failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing delete method with existing model finished.")

    print(f"{datetime.datetime.now()} Testing delete method with non-existing model_id fake_model")
    response = requests.delete(
        URL+f"_plugins/_ml/models/fake_model", auth=AUTH, verify=False)
    try:
        assert response.json()["error"]["type"] == "status_exception",\
            f"Testing get method with non-existing model failed, " \
            f"expect 'status_exception', return {response.json()['error']['root_cause']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing delete method with non-existing model failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing delete method with non-existing model finished.")

    # Get task
    print(f"{datetime.datetime.now()} Testing get method with existing task_id "
          f"{san_check_id_dict['async_logi']['task_id']}.")
    response = requests.get(URL+f"_plugins/_ml/tasks/{san_check_id_dict['async_logi']['task_id']}",
                            auth=AUTH, verify=False)
    try:
        assert response.json()["model_id"] == san_check_id_dict['async_logi']['model_id'],\
            "Testing get method with existing task failed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing get method with existing task failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing get method with existing model finished.")

    print(f"{datetime.datetime.now()} Testing get method with non-existing task_id fake_task.")
    response = requests.get(
        URL+f"_plugins/_ml/tasks/fake_task", auth=AUTH, verify=False)
    try:
        assert response.json()["error"]["reason"] == "Fail to find task",\
            f"Testing get method with non-existing task failed, " \
            f"expect 'Fail to find task', return {response.json()['error']['root_cause']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing get method with non-existing task failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing get method with non-existing task finished.")

    # Search task
    print(f"{datetime.datetime.now()} Testing search API with querying all tasks")
    request_body = json.dumps({"query": {"match_all": {}}, "size": 1000})
    response = requests.post(URL+f"_plugins/_ml/tasks/_search",
                             data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["hits"]["total"]["value"] > 0,\
            f"Testing search API with querying all tasks failed, " \
            f"expect model number greater than 0, return 0, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing search API with querying all tasks failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing search API with querying all tasks finished.")

    print(f"{datetime.datetime.now()} Testing search API with querying AD tasks")
    request_body = json.dumps(
        {"query": {"bool": {"filter": [{"term": {"function_name": "BATCH_RCF"}}]}}})
    response = requests.get(URL+f"_plugins/_ml/tasks/_search",
                            data=request_body, headers=HEADERS, auth=AUTH, verify=False)
    try:
        assert response.json()["hits"]["total"]["value"] > 0,\
            f"Testing search API with querying AD tasks failed, " \
            f"expect task number greater than 0, return 0, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing search API with querying AD only tasks failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing search API with querying kmeans tasks finished.")

    # Delete task
    print(f"{datetime.datetime.now()} Testing delete method with existing task_id "
          f"{san_check_id_dict['async_logi']['task_id']}.")
    response = requests.delete(URL+f"_plugins/_ml/tasks/{san_check_id_dict['async_logi']['task_id']}",
                               auth=AUTH, verify=False)
    try:
        assert response.json()["result"] == "deleted",\
            "Testing delete method with existing task failed, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing delete method with existing task failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing delete method with existing model finished.")

    print(f"{datetime.datetime.now()} Testing delete method with non-existing task_id fake_task.")
    response = requests.delete(
        URL+f"_plugins/_ml/tasks/fake_task", auth=AUTH, verify=False)
    try:
        assert response.json()["error"]["reason"] == "Fail to find task",\
            f"Testing delete method with non-existing task failed, " \
            f"expect 'Fail to find task', return {response.json()['error']['root_cause']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing delete method with non-existing task failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing delete method with non-existing task finished.")

    # Stats
    print(f"{datetime.datetime.now()} Testing stats API with getting all stats")
    response = requests.get(URL+f"_plugins/_ml/stats", auth=AUTH, verify=False)
    try:
        assert len(response.json()["nodes"]) > 0,\
            f"Testing stats API with getting all stats failed, " \
            f"expect node number greater than 0, return 0, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing stats API with getting all stats failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing stats API with getting all stats finished.")

    print(f"{datetime.datetime.now()} Testing stats API with getting specific stats of all nodes")
    response = requests.get(
        URL+f"_plugins/_ml/stats/ml_request_count", auth=AUTH, verify=False)
    try:
        assert any(dict(k)['ml_request_count'] for k in response.json()["nodes"].values()),\
            f"Testing stats API with getting specific stats of all nodes failed, " \
            f"expect some nodes'ml_request_count greater than 0, return 0, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing stats API with getting specific stats of all nodes failed '
              f'with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing stats API with getting specific stats of all nodes finished.")

    print(f"{datetime.datetime.now()} Testing stats API with getting stats of one specific node")
    sample_node_id = [k for k, v in response.json()["nodes"].items() if v['ml_request_count'] > 0][0]
    response = requests.get(
        URL+f"_plugins/_ml/{sample_node_id}/stats", auth=AUTH, verify=False)
    try:
        assert len(response.json()["nodes"]) == 1 and [k for k in response.json()["nodes"]][0] == sample_node_id,\
            f"Testing stats API with getting stats of one specific node failed, " \
            f"expect getting 1 node with id {sample_node_id}, return {len(response.json()['nodes'])} node " \
            f"with id {response.json()['nodes']}, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing stats API with getting all stats failed with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing stats API with getting stats of one specific node finished.")

    print(f"{datetime.datetime.now()} Testing stats API with getting specific stats of one specific node")
    response = requests.get(URL+f"_plugins/_ml/{sample_node_id}/stats/ml_request_count",
                            auth=AUTH, verify=False)
    try:
        assert response.json()['nodes'][sample_node_id].get('ml_request_count') is not None,\
            f"Testing stats API with getting specific stats of one specific node, " \
            f"fail to get a node with id {sample_node_id} and entry ml_request_count, please check."
    except KeyError:
        print(f'{datetime.datetime.now()} '
              f'Testing stats API with getting specific stats of one specific node failed '
              f'with status code {response.status_code}')
    print(f"{datetime.datetime.now()} Testing stats API with getting specific stats of one specific node finished.")

    # Security tests
    if AUTH is not None:
        security_test(san_check_id_dict)

    # TODO: can return an error counter here for clearer output
    print(f"{datetime.datetime.now()} Sanity check finished.")


if __name__ == '__main__':
    main()
