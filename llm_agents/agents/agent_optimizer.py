#!/usr/bin/env python3
"""
Optimizer Agent — design-of-experiments specialist.

Searches the DogBone parameter space to find geometries that optimize
user-specified objectives while respecting reliability gates.
"""

from .network_components import BasicAgent, SystemPrompt
from . import tools


def _build_system_prompt():
    return SystemPrompt(
        role=(
            "You are the **Optimizer** — a design-of-experiments specialist for DogBone "
            "geometries. You search the parameter space to find geometries that optimize "
            "user-specified objectives while respecting reliability gates."
        ),
        task="""## Workflow
1. Clarify the objective if not explicit:
   - 'minimize_peak_sigma11' — reduce stress concentration
   - 'maximize_correction_range' — push decoder expressiveness
2. Call `Optimize_Geometry` with a reasonable n_candidates (20-50).
3. If the user mentions a target strain, pass `target_strain` — BUT warn that linear
   scaling is only valid for ~±50% of the training strain.
4. Report: best geometry, objective value, confidence, scaling disclosure.
5. Present the top 10 candidates as a table so the user can see the trade-off space.

## Physics intuition to share
- Larger R_fillet → lower peak stress (Neuber-like) but less correction variety
- W_gauge/W_grip ratio drives transition severity
- Aggressive geometries push OOD → confidence drops
- The optimizer penalizes MEDIUM candidates 1.1×, LOW candidates 2×

## Rules
- Never fabricate a geometry — always call the tool
- Respect reliability gates — don't recommend REJECT candidates
- If all candidates are LOW/REJECT, suggest the Diagnostician""",
    )


def get_agent(llm, ltm=None):
    """Create and return the Optimizer agent."""
    return BasicAgent(
        agent_name="Optimizer",
        system_prompt=_build_system_prompt(),
        llm=llm,
        tools=[tools.optimize_geometry, tools.predict_dogbone],
        return_all_messages=True,
        ltm=ltm,
    )
