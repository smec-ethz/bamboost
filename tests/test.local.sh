#!/bin/bash

SKIP_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	-s | --skip)
		SKIP_BUILD=true
		shift
		;;
	*)
		PYTHON_VERSION=$1
		shift
		;;
	esac
done

# Check if a Python version is provided
if [ -z "$PYTHON_VERSION" ]; then
	echo "Please provide a Python version as an argument (e.g. ./test.sh 3.10 or ./test.sh -s 3.10)"
	exit 1
fi

IMAGE_NAME="bamboost-py${PYTHON_VERSION}"

if [ "$SKIP_BUILD" = false ]; then
	podman build -t $IMAGE_NAME -f test.Dockerfile --build-arg=PYTHON_VERSION=$PYTHON_VERSION
fi

podman run --rm -v $(pwd):/mnt --entrypoint /bin/bash $IMAGE_NAME:latest -c "
    source .venv/bin/activate && \
    cd /mnt && \
    uv sync --active --group test && \
    pytest tests
"
