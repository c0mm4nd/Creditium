# Ethereum Proof of Authority

This repository is an example setup allowing an arbitrary number of sealer nodes to
participate a Proof of Authority (PoA) concensus over an Ethereum network.

The provided genesis block defines a mining period of 5 seconds, and should allow
the network to systain production at this frequency without any issue.

To start the example cluster, simply run `docker-compose up`. By default, only the
first three nodes defined in the addresses array will be allowed to seal. To allow a
new node to participate the network, that node needs to be proposed and voted in
by 50% + 1 node. To do this, you can run in sequence:

```bash
docker-compose exec -T sealer-one geth --exec 'clique.propose("0x0d1d4e623d10f9fba5db95830f7d3839406c6af2", true)' attach
docker-compose exec -T sealer-two geth --exec 'clique.propose("0x0d1d4e623d10f9fba5db95830f7d3839406c6af2", true)' attach
```

At this point, the fourth node should be allowed to seal. To allow the fifth node to seal,
you can run the following commands:

```bash
docker-compose exec -T sealer-one geth --exec 'clique.propose("0x2932b7a2355d6fecc4b5c0b6bd44cc31df247a2e", true)' attach
docker-compose exec -T sealer-two geth --exec 'clique.propose("0x2932b7a2355d6fecc4b5c0b6bd44cc31df247a2e", true)' attach
docker-compose exec -T sealer-four geth --exec 'clique.propose("0x2932b7a2355d6fecc4b5c0b6bd44cc31df247a2e", true)' attach
```

At this point, the fifth node should start being able to seal blocks as well.

At any time, you can retrieve the list of authorized sealers by running:

```bash
docker-compose exec -T node geth --exec 'clique.getSigners()' attach
```

To vote a node out of the network, you can use the same commands as described before and simply replace `true` by false.
