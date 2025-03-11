#!/bin/sh
set -e

echo "Updating docsite repository..."
cd /website && git pull origin dev/next

echo "Replacing docs..."
rm -rf /website/content/docs
cp -r /bamboost/docs/content /website/content/docs

echo "Activating virtual environment and installing Python dependencies..."
. .venv/bin/activate
uv pip install /bamboost -U --no-deps
uv pip install /website/fumapy -U

echo "Installing npm dependencies and building the site..."
npm ci
npm run create-source
npm run build

echo "Copying build output..."
cp -r /website/out /bamboost/public

echo "Build complete."
