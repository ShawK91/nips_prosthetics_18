import yaml
import os
import time
import copy
from kubernetes import client, config

# from kubernetes import K8sConfig

# cfg_default = K8sConfig()

NAMESPACE = "ailab-users-zdwiel"

config.load_kube_config()
batch = client.BatchV1Api()
job_dict = yaml.load(open("gpu_job.yaml"))
# job_dict["spec"]["template"]["spec"]["containers"][0]["command"] = 'python main.py  -seed_pop f -save_folder /mnt/test -pop_size 200'
# job_dict["spec"]["template"]["spec"]["containers"][1]["command"] = 'python full_pg.py -seed_policy f -save_folder /mnt/test -num_workers 90'
# print(job_dict)
# exit()
containers = copy.copy(job_dict["spec"]["template"]["spec"]["containers"])
for container in containers:
    job_dict["spec"]["template"]["spec"]["containers"] = [container]
    response = batch.create_namespaced_job(body=job_dict, namespace=NAMESPACE)
    # print(response.spec.containers)
    print(response.metadata.name)

# # # delete_namespaced_job
# # for _ in range(10):
# #     print(batch.read_namespaced_job(response.metadata.name, namespace=NAMESPACE))
# #     time.sleep(1)
# time.sleep(5)
# os.system("./kubetail {name}".format(name=response.metadata.name))
