# ICAP Project Notes

## Project Overview
**ICAP (Intelligent Collective Autonomous Phenomenology)** is an experimental robotics research platform that studies self-organized robotic behavior using pheromone-based mechanisms. The project implements a distributed optimization system that trains robots to learn collective behaviors through swarm intelligence, specifically focusing on food foraging and transport tasks.

### Research Goals
- Train robots to develop self-organized behaviors through collective intelligence
- Evaluate pheromone mechanisms' contribution to robotic swarm behavior  
- Analyze relationships between environmental parameters and emergent behaviors
- Create an explainable model for understanding collective robotics

## Memories & Insights

### Development Practices
- `.venv`等の中は必要に合わせて参考にするのはいい. 例えば関数の使い方とか. でもリファクタリングとかでその中身まで調べるのは無駄すぎるから避けて

### Critical Constraints & Requirements
- **MuJoCo JAX Ray Casting**: mjx.ray() requires body_id to be static (compile-time constant). This forces architectural decisions like pre-compiled function registries rather than dynamic batching approaches. Never suggest "optimizations" that violate this constraint.
- **JAX Static Arguments**: Always verify static_argnames requirements before suggesting code changes to JAX-compiled functions.
- **Performance vs Thread Safety Trade-offs**: The codebase uses global state for pre-compiled JAX functions (e.g., _EMIT_RAYS_FUNCTIONS). While this creates potential race conditions in concurrent scenarios, it's the optimal performance approach for research simulations. Do NOT over-engineer thread-safe solutions that sacrifice performance unless explicitly required for concurrent usage.

## Architecture & Performance
[... rest of the existing content remains unchanged ...]