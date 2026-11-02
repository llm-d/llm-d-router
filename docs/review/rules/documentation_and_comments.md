# Documentation and comments

## Colocate plugin READMEs and link to canonical upstream guides
Document each plugin's parameter schema, code-verified defaults, and minimal YAML example in a colocated `README.md` inside its package under `pkg/epp/framework/plugins`. Link to upstream `llm-d/llm-d` architecture guides rather than duplicating them, using file or symbol links without source line numbers.

## Synchronize docstrings, diagrams, and configuration examples across review rounds
Update affected package docstrings, architecture diagrams, and YAML configuration examples whenever code or types change during review or rebase, and omit hardcoded plugin or feature counts.
