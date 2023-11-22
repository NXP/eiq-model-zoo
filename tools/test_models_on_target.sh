#!/usr/bin/env bash
# Copyright 2022-2023 NXP
# SPDX-License-Identifier: MIT

set -e

usage() {
    echo "This script runs all compatible .tflite models on the specified target board."
    echo "Usage ./tools/test_models_on_targets.sh -b <imx8mp|imx93> -a <IP ADDRESS>"
    echo "Flags:"
    echo "      -b target board. Supported targets: imx8mp, imx93"
    echo "      -a target IP address"
}

exit_error() {
    usage
    exit 1
}

while getopts b:a: flag
do
    case "${flag}" in
        b) board=${OPTARG};;
        a) address=${OPTARG};;
        *) exit_error;;
    esac
done

if [ "${board}" == "imx8mp" ]; then
    TFLITE_FILES=$(find . -name "*.tflite" | grep -v "vela")
    DELEGATE_PATH="/usr/lib/libvx_delegate.so"
elif [ "${board}" == "imx93" ]; then
    TFLITE_FILES=$(find . -name "*.tflite" | grep "vela")
    DELEGATE_PATH="/usr/lib/libethosu_delegate.so"
else
    exit_error
fi

if [ -z ${address+x} ]; then
    exit_error
fi

echo "Found the following tflite models:"
echo "${TFLITE_FILES}"

TMP_DIR=$(ssh root@"${address}" mktemp -d)
echo "Created temp dir on board ""${TMP_DIR}"

echo "Copying models on target..."
scp -q $TFLITE_FILES root@"${address}":"${TMP_DIR}"/
echo "DONE."

BENCHMARK_MODEL=$(ssh root@"${address}" "ls /usr/bin/tensorflow-lite-*/examples/benchmark_model")

for model in ${TFLITE_FILES}; do
    name=$(basename "${model}")
    echo -e "\n\n=========================================================\n\n"
    echo "Testing ${name}"
    ssh root@"${address}" "${BENCHMARK_MODEL}" --graph="${TMP_DIR}"/"${name}" --external_delegate_path="${DELEGATE_PATH}" | grep "Inference timings"
    ssh root@"${address}" "${BENCHMARK_MODEL}" --graph="${TMP_DIR}"/"${name}" | grep "Inference timings"
done

ssh root@"${address}" rm -rf "${TMP_DIR}"
