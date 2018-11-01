REGISTRY=nervana-dockrepo01.fm.intel.com:5001/
IMAGE=prosthetics_${USER}
EXPERIMENTS=prosthetics_${USER}

.PHONY: build run

BUILD_ARGUMENTS=
RUN_ARGUMENTS=
ifdef http_proxy
	BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
	RUN_ARGUMENTS+=--env http_proxy=$(http_proxy)
endif

ifdef https_proxy
	BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
	RUN_ARGUMENTS+=--env https_proxy=$(https_proxy)
endif

CONTEXT = $(dir $(dir $(shell pwd)))
CONTEXT = $(realpath ..)

DOCKER := $(shell command -v nvidia-docker 2> /dev/null)

ifndef DOCKER
	DOCKER = nvidia-docker
endif

RUN_ARGUMENTS+=--rm
RUN_ARGUMENTS+=-v /tmp/shared:/shared

clean:
	@rm -f .*.swp .*.swo
	@rm -f *.pyc

build:
	${DOCKER} build -f=Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT}

run: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE}

run_pg: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} python /root/src/full_pg.py

shell: build
	${DOCKER} run ${RUN_ARGUMENTS} -it ${IMAGE} /bin/bash

test: build
	${DOCKER} run ${RUN_ARGUMENTS} -t ${IMAGE} py.test tests

push: build
	${DOCKER} tag ${IMAGE} ${REGISTRY}${IMAGE}
	${DOCKER} push ${REGISTRY}${IMAGE}

run_kubernetes: push
	kubectl create -f gpu_job.yaml
