# PI-GINOT — Physics-Informed Geometry-Informed Neural Operator Transformer

A physics-informed neural operator for predicting displacement and stress fields on **parametric DogBone tensile specimens** under finite-strain Neo-Hookean hyperelasticity.

The model learns the solution operator *without any simulation data* — training is driven entirely by PDE residuals (equilibrium, traction boundary conditions, and section-force consistency).

## Architecture

PI-GINOT follows a branch–trunk neural operator design:

| Component | Role |
| --- | --- |
| **GeometryEncoder** (branch) | Encodes a boundary point cloud via FPS, ball-query grouping, and self-attention into latent geometry tokens |
| **PhysicsDecoder** (trunk) | Maps query coordinates + geometry latent to displacement via NeRF positional encoding, cross-attention, and FiLM conditioning |
| **Hard BC layer** | Enforces Dirichlet BCs exactly: u(0,y)=0, u(L,y)=u_delta, v(x,0)=0 |

**Encode-once pattern**: the geometry is encoded once per specimen; the decoder is then called separately on interior, traction-free, and partial-traction point sets for the physics loss.

### Key hyperparameters

| Parameter | Value |
| --- | --- |
| Embedding dimension | 64 |
| Encoder self-attention layers | 3 |
| Decoder cross-attention layers | 6 |
| Attention heads | 4 |
| NeRF PE max degree | 6 |
| Boundary PC size (encoder input) | 320 points |
| Interior collocation points | 4000 per geometry |

## Physics

- **Constitutive model**: compressible Neo-Hookean hyperelasticity
- **Stress measure**: 1st Piola–Kirchhoff stress P (reference configuration)
- **Equilibrium**: Div(P) = 0 enforced via AD on interior collocation points
- **Traction BCs**: P · N = 0 on gauge top and fillet arc; partial traction on symmetry planes
- **Section resultant**: axial force consistency N(x) = const across vertical slices
- **det(F) barrier**: penalises local element inversion during training
- **Stress state**: plane stress (configurable to plane strain)

### Material

| Property | Value |
| --- | --- |
| Young's modulus E | 760 MPa |
| Poisson's ratio ν | 0.23 |
| Unit system | mm, N, MPa |

## Geometry

Quarter-model DogBone specimen with double symmetry (x=0 and y=0). The geometry is parameterised by four variables sampled uniformly during training:

| Parameter | Range | Unit |
| --- | --- | --- |
| L_total (specimen length) | 40–70 | mm |
| W_grip (grip width) | 16–26 | mm |
| W_gauge (gauge width) | 6–14 | mm |
| R_fillet (fillet radius) | 8–20 | mm |

Boundary and interior points are generated analytically (no mesher required). A fixed bank of 128 training and 24 validation geometries is used, with collocation points resampled each epoch for training.

## Agentic AI Layer

The project includes a multi-agent conversational interface built with **LangGraph** and served via **Streamlit** (deployed as a Databricks App). It provides natural-language access to the trained PI-GINOT model through four specialist agents coordinated by a central router.

### Agent Architecture

The system is structured as a LangGraph `StateGraph` with the following flow:

```
User → ReduceMessages → CacheCheck → Router → [Specialist Agent] → CacheStore → Response
```

**Router** — An LLM-based classifier that inspects the user query and delegates to the appropriate specialist using structured output. Stays with the current agent unless the user explicitly switches context.

**Specialist Agents** — Each is a `BasicAgent` subgraph implementing a ReAct tool loop (LLM → tool calls → LLM → ... → final answer):

| Agent | Responsibility |
| --- | --- |
| **Predictor** | Runs reliability-aware predictions on arbitrary DogBone geometries, interprets stress/displacement fields with physics context, manages a geometry library |
| **Optimizer** | Searches the parameter space to find geometries minimizing peak stress or maximizing correction range, reports trade-off tables |
| **Diagnostician** | Diagnoses low-confidence predictions, identifies rejection reasons, suggests refinement strategies |
| **Reporter** | Generates narrative reports with physics interpretation from prior prediction/optimization data |

### Memory and Caching

| Component | Purpose |
| --- | --- |
| **Semantic Cache** | Embedding-based similarity matching (cosine threshold 0.92) avoids re-running inference for semantically equivalent queries |
| **Long-Term Memory** | Persists user preferences, named geometries, and agent interaction history across sessions |
| **Short-Term Memory** | Tracks session activity and thread state for conversation continuity |
| **Message Reduction** | Trims and summarizes old conversation turns to stay within context limits |

### Streamlit Pages

| Page | Description |
| --- | --- |
| **Analyst** | Multi-agent chat interface with rich output rendering (field plots, diagnostics cards, optimization traces, tables) |
| **Design Studio** | Interactive geometry explorer with live prediction, reliability heatmap, comparison mode, and a "Race the Optimizer" challenge |
| **Demo** | Showcase mode for presentations |

## Deployment

### Databricks App (production)

The app is deployed as a Databricks App via `app.yaml`. Authentication to the Databricks model serving endpoint is handled automatically.

```bash
databricks apps deploy pi-ginot-studio --source-code-path .
```

Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are managed through Databricks secret scopes referenced in `app.yaml`.

### External / Self-hosted (Docker)

For deployment outside Databricks (local, AWS, GCP, Azure VMs, etc.):

```bash
# 1. Configure secrets
cp .env.example .env
# Edit .env with your API keys

# 2. Place model checkpoint
#    Ensure checkpoints/best.pt exists (or the app runs in chat-only mode)

# 3. Build and run
docker compose up --build
```

The app will be available at `http://localhost:8501`.

Alternatively, without Docker:

```bash
pip install -r requirements-external.txt
export OPENAI_API_KEY=sk-...
streamlit run llm_agents/Home.py
```

**Key differences from Databricks deployment:**
- Uses `requirements-external.txt` (excludes `databricks-langchain`)
- The "Databricks Endpoint" LLM option is automatically hidden when `databricks-langchain` is not installed
- Secrets are provided via environment variables (`.env` file) instead of Databricks secret scopes
- Model checkpoint is mounted as a Docker volume rather than residing in the workspace

## Training

```
python main.py --mode train --epochs 1500 --device cuda
python main.py --mode train --resume checkpoints/last.pt
```

**Loss** = w_eq · L_eq + w_trac · L_trac + w_part · L_part + w_barrier · L_barrier + w_res · L_resultant

| Setting | Value |
| --- | --- |
| Optimizer | Adam, lr=1e-4 |
| LR scheduler | ReduceLROnPlateau (factor=0.7, patience=10) |
| Gradient clipping | norm ≤ 1.0 |
| Batch size | 8 geometries per epoch |
| Adaptive loss weighting | EMA-based, starts at epoch 200 |
| Checkpoint gating | latent-swap sensitivity + section-force CV |

## Evaluation

Evaluate all validation geometries and produce per-specimen field plots:

```
python evaluate_all_val.py
python evaluate_all_val.py --checkpoint checkpoints/best.pt --out_dir results/
```

Generate an animated showcase (GIF/MP4) of predictions on random specimens:

```
python showcase_gif_v2.py --ckpt checkpoints/last.pt --n_triplets 12 --fps 1.0
```

## Project structure

```
PI-GINOT-DogBone-AgenticAI/
├── config.py                        # Unified configuration
├── main.py                          # Training entry point
├── evaluate_all_val.py              # Validation evaluation script
├── showcase_gif_v2.py               # Animated showcase generator
├── write_out.py                     # Result export utilities
├── app.yaml                         # Databricks App deployment config
├── Dockerfile                       # External container deployment
├── docker-compose.yml               # Docker orchestration
├── requirements.txt                 # Databricks deployment dependencies
├── requirements-external.txt        # External deployment dependencies
├── .env.example                     # Environment variable template
│
├── models/
│   ├── pi_ginot.py                  # Top-level model (encoder + decoder)
│   ├── geometry_encoder.py          # Boundary point cloud encoder
│   ├── physics_decoder.py           # Cross-attention decoder with hard BCs
│   └── modules/
│       ├── transformer.py           # Attention blocks and MLPs
│       ├── point_encoding.py        # Point cloud feature extraction
│       ├── point_position_embedding.py  # NeRF positional encoding
│       └── pointnet2_utils.py       # FPS, ball query, grouping
│
├── geometry/
│   ├── parametric_dogbone.py        # Analytical DogBone geometry generator
│   └── collocation.py              # Collocation point sampling
│
├── training/
│   ├── trainer.py                   # Multi-geometry training loop
│   └── curriculum.py                # Curriculum scheduler (currently disabled)
│
├── physics/
│   ├── losses.py                    # Combined physics loss function
│   ├── neo_hookean.py               # Neo-Hookean constitutive model
│   └── equilibrium.py              # Equilibrium residual via AD
│
├── agent/                           # Reliability gating and verification
│   ├── agent.py                     # Inference orchestrator with gates
│   ├── inference.py                 # Model inference wrapper
│   ├── reliability.py               # Confidence scoring
│   ├── verification.py              # Section-force and residual checks
│   ├── gates.py                     # Accept/reject logic
│   ├── refinement.py                # Collocation refinement strategies
│   └── health_check.py             # Model health diagnostics
│
├── llm_agents/                      # Agentic AI interface
│   ├── Home.py                      # Streamlit entry point
│   ├── requirements.txt             # Python dependencies for the app
│   ├── pages/
│   │   ├── 1_Analyst.py            # Multi-agent chat UI
│   │   ├── 2_Design_Studio.py      # Interactive geometry explorer
│   │   └── 3_Demo.py               # Showcase / demo mode
│   └── agents/
│       ├── network_components.py    # State, BasicAgent, AgentRouter, reducers
│       ├── tools.py                 # LangChain tools wrapping PI-GINOT inference
│       ├── agent_predictor.py       # Predictor specialist
│       ├── agent_optimizer.py       # Optimizer specialist
│       ├── agent_diagnostician.py   # Diagnostician specialist
│       ├── agent_reporter.py        # Reporter specialist
│       ├── semantic_cache.py        # Embedding-based query cache
│       ├── long_term_memory.py      # Persistent user memory
│       ├── short_term_memory.py     # Session state tracking
│       ├── cache_nodes.py           # LangGraph cache check/store nodes
│       └── callbackhandler.py       # Streamlit streaming callback
│
└── checkpoints/                     # Saved weights and training history
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, Matplotlib, Pillow, imageio
- LangChain, LangGraph, Streamlit (for the agentic layer)
- OpenAI or Anthropic API key (for LLM-based agents)

## License

This project is licensed under the [MIT License](LICENSE).
