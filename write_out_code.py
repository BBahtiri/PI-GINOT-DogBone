#!/usr/bin/env python3
"""
Write out the entire PI-GINOT Multi-Agent codebase into a single file.

Reads all Python source files from the project and concatenates them
into PROJECT_CODE.txt with clear section headers for each file.

Usage:
    python write_out_code.py
    python write_out_code.py --output my_dump.txt
    python write_out_code.py --include-all  # includes training, physics, etc.
"""

import os
import argparse
from pathlib import Path


# Ordered list of files to include (core multi-agent architecture)
LLM_AGENTS_FILES = [
    "llm_agents/__init__.py",
    "llm_agents/Home.py",
    "llm_agents/pages/1_Analyst.py",
    "llm_agents/pages/2_Design_Studio.py",
    "llm_agents/pages/3_Demo.py",
    "llm_agents/agents/__init__.py",
    "llm_agents/agents/network_components.py",
    "llm_agents/agents/tools.py",
    "llm_agents/agents/callbackhandler.py",
    "llm_agents/agents/agent_predictor.py",
    "llm_agents/agents/agent_optimizer.py",
    "llm_agents/agents/agent_diagnostician.py",
    "llm_agents/agents/agent_reporter.py",
    "llm_agents/agents/short_term_memory.py",
    "llm_agents/agents/long_term_memory.py",
    "llm_agents/agents/semantic_cache.py",
    "llm_agents/agents/cache_nodes.py",
    "llm_agents/requirements.txt",
]

# Additional project files (deterministic PI-GINOT stack)
INFRASTRUCTURE_FILES = [
    "config.py",
    "main.py",
    "write_out.py",
    "write_out_code.py",
    "showcase_gif_v2.py",
    "evaluate_all_val.py",
    "agent/__init__.py",
    "agent/constants.py",
    "agent/schemas.py",
    "agent/gates.py",
    "agent/verification.py",
    "agent/reliability.py",
    "agent/inference.py",
    "agent/refinement.py",
    "agent/health_check.py",
    "agent/agent.py",
    "models/__init__.py",
    "models/geometry_encoder.py",
    "models/physics_decoder.py",
    "models/pi_ginot.py",
    "models/modules/__init__.py",
    "models/modules/point_encoding.py",
    "models/modules/pointnet2_utils.py",
    "models/modules/transformer.py",
    "models/modules/point_position_embedding.py",
    "physics/__init__.py",
    "physics/neo_hookean.py",
    "physics/equilibrium.py",
    "physics/losses.py",
    "geometry/__init__.py",
    "geometry/parametric_dogbone.py",
    "geometry/collocation.py",
    "training/__init__.py",
    "training/trainer.py",
    "training/curriculum.py",
]


def write_out(output_path: str, include_all: bool = False):
    """Concatenate source files into a single output file."""
    project_root = Path(os.getcwd())

    files_to_include = list(LLM_AGENTS_FILES)
    if include_all:
        files_to_include += INFRASTRUCTURE_FILES

    separator = "=" * 80

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"{separator}\n")
        out.write("PI-GINOT Multi-Agent System — Full Source Code Dump\n")
        out.write(f"{separator}\n")
        out.write(f"Files included: {len(files_to_include)}\n")
        out.write(f"Generated from: {project_root}\n")
        out.write(f"{separator}\n\n")

        for rel_path in files_to_include:
            full_path = project_root / rel_path
            out.write(f"\n{'#' * 80}\n")
            out.write(f"# FILE: {rel_path}\n")
            out.write(f"{'#' * 80}\n\n")

            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            else:
                out.write(f"# [FILE NOT FOUND: {full_path}]\n")

            out.write("\n")

    print(f"Written {len(files_to_include)} files to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump PI-GINOT source to single file")
    parser.add_argument("--output", "-o", default="PROJECT_CODE.txt",
                        help="Output file path (default: PROJECT_CODE.txt)")
    parser.add_argument("--include-all", action="store_true",
                        help="Include full project (agent, models, physics, training)")
    args = parser.parse_args(["--include-all"])

    write_out(args.output, include_all=args.include_all)
