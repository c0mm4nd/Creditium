#!/usr/bin/env ash
. ./scripts/functions.sh

bootnode=$(getent hosts bootnode | awk '{print $1}')
datadir=/root/.ethereum
etherbase=$1
addresses="0x1596220ad0adebb0090838cb6d99ff7737b683e6,0x402e6377bbed775a3b3ebc7c3df9a21bf61ae49f,0x356Cf535FC1f1C4dD1a4C4f4c261684bcd3e1bcb,0x59Ef5B8b24cE3e00Ad0345cD2d47f96947c3e2c3,0x4B54772Cc9e233B4fe1F04A65a994aF40A6834ac,0xF7e5800E52318834E8689c37dCCCD2230427a905,0xC9c4496508E92A9FCB0Ffc8Cb6363f910C7E8AE3,0x6c319A49787d10a6e0c72a25DB06cc555370c4DD,0x9E9581516679F57Aa7eb81D278841DF6aB93902B,0x71b3D7405080197fC03cA82bCDd1764F1e14ABf2,0x0Cbbb79B02449ea575F6185dd3C541E9ab8d8182,0x563A5fC36b990D68bBCaAA206C281BfEc31134AB"

mkdir -p $datadir/keystore
cp -R /keystore $datadir
prepareDatadir $datadir $etherbase

geth --datadir $datadir \
     --miner.etherbase $etherbase \
     --networkid 1231312313142 \
     --miner.gasprice "1" \
     --miner.gaslimit "0x59a5380" \
     --http --http.addr '0.0.0.0' --http.port 8545 --http.corsdomain '*' --http.api 'admin,db,eth,miner,net,web3,personal,txpool' --http.vhosts "*" \
     --ws --ws.addr '0.0.0.0' --ws.port 8546 --ws.origins '*' --ws.api 'admin,db,eth,miner,net,web3,personal,txpool'  \
     --bootnodes "enode://f94118749beb981da38b82ab6be7b00dc0082783d698080fd0ae45a2c3d42f1ce74cbb153ffcfb1379b64235605bfff43f85b112032ddd9685ad2ab88735e1b1@${bootnode}:30301" \
     --unlock $addresses \
     --allow-insecure-unlock \
     --password "/dev/null" \
     --verbosity 3
