# Solution Notes — Route Inspection (Chinese Postman)

## 1. Approach
The problem is an undirected Chinese Postman Problem: find an edge-covering walk with the lowest total weight.

The basic pipeline:
1. Find all odd-degree vertices ($V_{\text{odd}}$).
2. Compute shortest paths between them.
3. Find a minimum-weight perfect matching (MWPM) to pick which paths to duplicate.
4. Extract the Eulerian walk using Hierholzer's algorithm.

This works fine for `hard_to_choose` (0 duplication needed) and small graphs.

## 2. Scaling on Paris
The `paris_map` instance has 7,318 odd vertices.
- A full distance matrix between all odd vertices means computing about 26.8M pairs, which runs out of memory and time.
- I tried $k$-nearest-neighbor pruning to keep it small, but testing $k \in \{3, 6, 10\}$ dropped the duplication cost from 5,056 to 5,005 to 5,003. This showed $k$-NN was just a heuristic without a guarantee.
- Instead of matching on a dense distance graph, the exact problem is finding a **Minimum Weight $T$-join** ($T = V_{\text{odd}}$) on the original sparse graph.
- Using `PyMatching`'s sparse blossom solver, the exact optimal $T$-join runs in **0.16 s** directly on the graph, giving a duplication cost of **5,003**. Total runtime including graph loading and Hierholzer is about **1.0 s**.

## 3. Edge Cases
- **`islands.txt` (Disconnected graph):** A single continuous walk across separate components is impossible with the given edges. The solver splits the graph by connected component and returns one tour per component.
- **Open walk (`--open`):** Instead of checking all pairs of endpoints, we add two virtual vertices connected to all odd nodes with weight $M = \text{ecc}(r) + 1$. Because $M > \text{diam}/2$, the solver never uses more than two virtual edges and directly picks the best two endpoints.

## 4. Results

| Instance | Original Cost | Duplication Cost | Closed Walk | Open Walk (`--open`) | Runtime (Total) |
|---|---:|---:|---:|---:|---:|
| `hard_to_choose` | 1,200 | 0 | 1,200 | 1,200 | < 0.05 s |
| `islands` (3 components) | 1,896 | 20 | 1,916 | 1,914 | < 0.05 s |
| `paris_map` | 22,924 | **5,003** | **27,927** | **27,918** | **~1.0 s** *(T-join: 0.16 s)* |

## 5. Tooling Disclosure
I am in the middle of a code sprint at work, so I did not have much time to allocate to the project.
I chose the formulation and the overall approach (using a sparse $T$-join for Paris, component splitting, and the virtual-vertex setup for open walks). I used Claude Code to write some of the code and tests, and to run the benchmarks.