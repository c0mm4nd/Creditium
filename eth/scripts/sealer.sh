#!/usr/bin/env ash
. ./scripts/functions.sh

bootnode=$(getent hosts bootnode | awk '{print $1}')
datadir=/root/.ethereum
etherbase=$1

mkdir -p $datadir/keystore
cp /keystore/$etherbase.json $datadir/keystore
prepareDatadir $datadir $etherbase

geth --datadir $datadir \
     --miner.etherbase $etherbase \
     --networkid 1231312313142 \
     --miner.gasprice "1" \
     --miner.gaslimit "0x59a5380" \
     --bootnodes "enode://f94118749beb981da38b82ab6be7b00dc0082783d698080fd0ae45a2c3d42f1ce74cbb153ffcfb1379b64235605bfff43f85b112032ddd9685ad2ab88735e1b1@${bootnode}:30301" \
     --allow-insecure-unlock \
     --unlock $etherbase \
     --password "/dev/null" \
     --verbosity 2 \
     --mine
