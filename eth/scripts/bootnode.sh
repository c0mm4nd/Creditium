#!/usr/bin/env bash
. ./scripts/functions.sh

prepareDatadir /root/.ethereum 0x0Cbbb79B02449ea575F6185dd3C541E9ab8d8182
bootnode --nodekey /boot.key \
         --verbosity 1
