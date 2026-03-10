# 18 — Graphs and Networks

NetworkX-based graph analysis from construction through machine learning. All examples use ecological networks: stream monitoring connectivity, species interaction webs, dispersal corridors, and plant-pollinator bipartite networks.

---

## Notebooks

### `networkx_basics.ipynb`
DiGraph construction with node/edge attributes. Hierarchical stream network visualisation (node size = richness, edge width = flow). Node attribute DataFrames. Adjacency matrix (`to_numpy_array`). Graph properties: connectivity, diameter, longest path, all simple paths. Undirected species co-occurrence network with threshold filtering. Pitfalls: directed vs undirected, view objects vs lists, duplicate edges, spring layout seed, undirected-only algorithms on DiGraph.

### `centrality_community.ipynb`
Degree, betweenness, closeness, eigenvector, PageRank centrality. Top-node comparison across measures. Node-sized/coloured centrality visualisations. Louvain community detection (falls back to greedy modularity). Modularity Q and community size inspection. Richness vs betweenness scatter by community. Pitfalls: unnormalised betweenness, modularity without size check, weight attribute for community detection, betweenness vs degree for keystones, null model comparison.

### `shortest_paths_flow.ipynb`
Dijkstra shortest path with resistance weights. Least-resistance path visualisation. All-pairs shortest path distance matrix heatmap. Network diameter and most-isolated patch. Maximum flow + minimum cut on directed flow network. Critical node analysis: removal impact on average path length. Pitfalls: negative weights with Dijkstra, resistance vs geodesic distance, connectivity check before flow, bridge-node removal handling, capacity vs weight attribute.

### `network_statistics.ipynb`
Random (ER), scale-free (BA), small-world (WS) graph comparison. Clustering coefficient and average path length vs random baseline. Small-world index. Degree distribution histograms. Power-law exponent fit for BA network. Assortativity coefficient. Robustness simulation: random vs targeted node removal. Null model test: observed clustering vs degree-preserving rewiring. Pitfalls: disconnected graph for path length, clustering alone for small-world, power-law fitting by eye, assortativity significance, static degree sorting for targeted attack.

### `bipartite_temporal.ipynb`
Plant-pollinator bipartite network with `bipartite=0/1` node attributes. Bipartite-specific layout visualisation. Weighted plant projection via shared pollinators. Incidence matrix with nestedness sorting visualisation. Connectance calculation. Temporal network: seasonal connectivity snapshots. Aggregate network with edge persistence count. Pitfalls: unipartite methods on bipartite, unnormalised projection, temporal aggregation masking dynamics, connectance vs completeness, sorted-matrix nestedness fallacy.

### `graph_ml.ipynb`
Node feature engineering: raw attributes + degree, betweenness, eigenvector, clustering, neighbourhood mean richness. RandomForest node classification: features-only vs features+topology. Link prediction with Jaccard, Adamic-Adar, preferential attachment, common neighbours. Node2Vec random walks + Word2Vec embeddings, PCA visualisation. Manual 2-layer GCN: normalised adjacency message passing. Pitfalls: ignoring topology, easy negative link prediction, test-edge leakage in topological features, Node2Vec p/q defaults, GNN over-smoothing.

---

## Dependencies
```
networkx, numpy, pandas, matplotlib, sklearn, scipy
gensim (optional, for Node2Vec: pip install gensim)
python-louvain (optional: pip install python-louvain)
```
