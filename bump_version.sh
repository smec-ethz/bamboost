#!/bin/bash

# Description: Bump the version of the package
# Usage: ./bump_version.sh [major|minor|patch]

INIT_FILE="./bamboost/__init__.py"
PYPROJECT_FILE="./pyproject.toml"

# Get the current version
VERSION=$(grep -oP '__version__ = "\K[^"]+' $INIT_FILE)
MAJOR=$(echo $VERSION | cut -d. -f1)
MINOR=$(echo $VERSION | cut -d. -f2)
PATCH=$(echo $VERSION | cut -d. -f3)

# Get the increment (-1 if -r is passed)
if [ "$2" == "--reduce" ] || [ "$2" == "-r" ]; then
    INC=-1
else
    INC=1
fi

# Increment the version
case $1 in
    major)
        MAJOR=$((MAJOR + $INC))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + $INC))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + $INC))
        ;;
    *)
        echo "Usage: $0 [major|minor|patch] [--reduce|-r]"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"

# Update the version in the __init__.py file
sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" $INIT_FILE

# Update the version in the pyproject.toml file
sed -i "s/version = \".*\"/version = \"$NEW_VERSION\"/" $PYPROJECT_FILE

echo "Bumped version to $NEW_VERSION"
