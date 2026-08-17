# HNSW (Hierarchical Navigable Small World) Architecture

This diagram visualizes how the FAISS HNSW algorithm achieves O(log N) search speeds across massive datasets. It achieves this by structuring vectors in a multi-layered hierarchical graph (similar to a skip-list), allowing a search query to take "high-speed highways" at the top layer, and progressively zoom into denser "local streets" at the bottom layer.

## HNSW Layer Visualization Diagram

```mermaid
graph TD
    %% Styling Definitions
    classDef query fill:#e74c3c,stroke:#c0392b,stroke-width:3px,color:#fff;
    classDef topNode fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff;
    classDef midNode fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#fff;
    classDef baseNode fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff;

    Q((Search<br/>Query)):::query

    subgraph Layer 2: The Global Highway (Extremely Sparse)
        L2_A((Node A)):::topNode
        L2_B((Node B)):::topNode
        L2_A --- L2_B
    end

    subgraph Layer 1: Regional Roads (Medium Density)
        L1_A((Node A)):::midNode
        L1_B((Node B)):::midNode
        L1_C((Node C)):::midNode
        L1_D((Node D)):::midNode
        
        L1_A --- L1_B
        L1_A --- L1_C
        L1_B --- L1_D
        L1_C --- L1_D
    end

    subgraph Layer 0: Local Streets (Dense - Contains 100% of Data)
        L0_A((Node A)):::baseNode
        L0_B((Node B)):::baseNode
        L0_C((Node C)):::baseNode
        L0_D((Node D)):::baseNode
        L0_E((Node E)):::baseNode
        L0_F((Node Target)):::baseNode
        
        L0_A --- L0_B & L0_C & L0_E
        L0_B --- L0_D & L0_F
        L0_C --- L0_D & L0_E
        L0_D --- L0_F
        L0_E --- L0_F
    end

    %% Active Search Path Traversal (Red Dashed Lines)
    Q -.->|1. Enter Top Layer| L2_A
    L2_A -.->|2. Navigate & Drop Down| L1_C
    L1_C -.->|3. Navigate & Drop Down| L0_E
    L0_E -.->|4. Find Nearest Target| L0_F

    %% Vertical Hierarchical Structure Links (Faded)
    L2_A ~~~ L1_A
    L2_B ~~~ L1_B
    L1_A ~~~ L0_A
    L1_B ~~~ L0_B
    L1_C ~~~ L0_C
    L1_D ~~~ L0_D
```

## Presentation Talking Points for HNSW
* **The Layer Cake Approach:** HNSW builds a pyramid of graphs. The bottom layer (Layer 0) contains every single IT ticket. The top layer (Layer 2) contains only a few random "entry point" tickets.
* **The Search Path:** When a new ticket comes in, the AI drops into the top layer. Because the top layer has very few tickets, it can instantly hop to the closest one. It then drops down to the next layer and refines the search, eventually landing on the absolute closest ticket at the dense bottom layer.
* **The Speed Gain:** Instead of checking 1,000,000 tickets individually, the AI might only do 30 hops through the graph layers to find the exact match. This turns a linear $O(N)$ scan into a logarithmic $O(\log N)$ scan!
