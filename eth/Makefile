next-block:
	# 0x6c319A49787d10a6e0c72a25DB06cc555370c4DD 0x9E9581516679F57Aa7eb81D278841DF6aB93902B 0x0Cbbb79B02449ea575F6185dd3C541E9ab8d8182 0x563A5fC36b990D68bBCaAA206C281BfEc31134AB
	docker-compose exec -T sealer-one geth --exec 'clique.propose("0x9E9581516679F57Aa7eb81D278841DF6aB93902B", true)' attach
	docker-compose exec -T sealer-two geth --exec 'clique.propose("0x9E9581516679F57Aa7eb81D278841DF6aB93902B", true)' attach
	docker-compose exec -T sealer-three geth --exec 'clique.propose("0x9E9581516679F57Aa7eb81D278841DF6aB93902B", true)' attach
	docker-compose exec -T sealer-four geth --exec 'clique.propose("0x9E9581516679F57Aa7eb81D278841DF6aB93902B", true)' attach
