#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --model-store model-store --models my_tc=bert-ner.mar
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
