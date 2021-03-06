#! /usr/bin/env bash

set -e

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 TMP_DIR"
  echo ""
  echo "TMP_DIR must point to a writeable, empty, temporary directory."

  exit 1
fi

TMP_DIR="$(realpath "$1")"
DOXYGEN_AWESOME_DIR="$(dirname "$(realpath "$0")")"
REPO_URL="https://github.com/jothepro/doxygen-awesome-css"
REPO_DIR="${TMP_DIR}/doxygen-awesome-css"

CONTENT_URL="https://raw.githubusercontent.com/jothepro/doxygen-awesome-css"

mkdir -p "${TMP_DIR}"
git clone ${REPO_URL} "${REPO_DIR}" 1>&2
pushd "${REPO_DIR}" 1>&2

VERSION="$(git tag -l | sed -e '/^v[0-9]*\.[0-9]*\.[0-9]*$/!d' | sort -V | tail -n 1)"

popd 1>&2

if [[ -z "$VERSION" ]]
then
  exit 1
fi

for STYLESHEET in "doxygen-awesome.css" "doxygen-awesome-sidebar-only.css" "doxygen-awesome-darkmode-toggle.js" "doxygen-awesome-sidebar-only-darkmode-toggle.css"; do
  curl "${CONTENT_URL}/${VERSION}/${STYLESHEET}" \
       --output "${DOXYGEN_AWESOME_DIR}/${STYLESHEET}" \
       1>&2
done

echo "${VERSION}"
