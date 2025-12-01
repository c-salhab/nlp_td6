## Résultats bruts

chunk_size = 768, overlap = 192

MRR = 0.2737

nb_chunks = 153

chunk_size = 512, overlap = 128

MRR = 0.2803 (meilleur)

nb_chunks = 199

chunk_size = 512, overlap = 0

MRR = 0.2301

nb_chunks = 186

chunk_size = 384, overlap = 96

MRR = 0.1932

nb_chunks = 252

chunk_size = 256, overlap = 64

MRR = 0.2036

nb_chunks = 357

chunk_size = 256, overlap = 0

MRR = 0.1373 (pire)

nb_chunks = 313

## Analyse des résultats

-Effet de la taille des chunks:

Les chunks très petits (256) donnent les pires MRR :
256 / 0 -> 0.1373
256 / 64 -> 0.2036
L’information pertinente est souvent éclatée sur plusieurs chunks ce qui fait baisser le rang du premier chunk correct

En augmentant la taille vers 512 puis 768, le MRR s’améliore :
512 / 0 -> 0.2301
768 / 192 -> 0.2737
Des chunks plus longs contiennent des passages plus complets (intrigue, descriptions, etc.), ce qui facilite la récupération du bon segment.

-Effet du chunk_overlap

À taille de chunk fixe l’overlap a un impact important:
Pour chunk_size = 512 :
overlap 0 -> MRR = 0.2301
overlap 128 -> MRR = 0.2803

Pour chunk_size = 256 :

overlap 0 -> 0.1373
overlap 64 -> 0.2036
on observe la même tendance : l’overlap permet de rattraper un peu la perte liée à des chunks trop courts, sans toutefois atteindre la qualité des chunks plus grands


-Compromis précision/taille de l’index

L’augmentation de l’overlap augmente aussi le nombre de chunks :
512 / 0 -> 186 chunks
512 / 128 -> 199 chunks
256 / 64 -> 357 chunks 

Il y a donc un compromis :
Plus de chunks -> index plus gros et calculs de similarité plus coûteux.
Mais aussi meilleur MRR, car plus de chances de couvrir complètement les passages pertinents.

## Conclusion

Nos expérimentations montrent que les hyperparamètres de chunking ont un impact fort sur la qualité du retrieval dans un RAG sur des pages de films :
-Les chunks trop petits sans overlap (256 / 0) dégradent fortement le MRR (0.1373), l’information étant trop fragmentée.
-Une taille de chunk moyenne à grande (512–768) avec overlap significatif donne les meilleurs résultats.