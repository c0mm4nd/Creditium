.PHONY: up down

up:
	docker-compose up -d ipfs0 ipfs1 ipfs2
	docker-compose up -d cluster0 cluster1 cluster2

down:
	docker-compose down
	rm -rf data
