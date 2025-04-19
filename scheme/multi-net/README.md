# multi net

We create a neural network composed of multiple sub-networks, each of which is a simple neural network.

The output of each sub-network is selected by a gating network.

Actual networks are shown in the following figure.

```mermaid
flowchart TB
    Input["Input"]

    subgraph SubNets["&nbsp;"]
        direction TB

        subgraph SubNet1["&nbsp;"]
            S1_A[Linear] -->|Tanhshrink| S1_B[Liner]
        end

        Repeat["SubNets<br>. . . . . . . ."]
        style Repeat fill: transparent, stroke: transparent

        subgraph SubNetN["&nbsp;"]
            SN_A[Linear] -->|Tanhshrink| SN_B[Liner]
        end
    end
    style SubNets fill: transparent

    subgraph GatingNetwork["&nbsp;"]
        direction TB
        G_A[Linear] -->|Tanhshrink| G_B[Liner]
    end
    style SubNets fill: transparent
%% Give Input to sub-nets and gating net  
    Input --> S1_A
    Input --> SN_A
    Input --> G_A
%% Stack the output of sub-nets
    S1_B -->|Tanh| Stack((Stack<br>dim=1))
    SN_B -->|Tanh| Stack
%% Gating network
    Stack --> Dot((Dot))
    G_B -->|Softmax| Dot
%% Output
    Dot --> Output
```
