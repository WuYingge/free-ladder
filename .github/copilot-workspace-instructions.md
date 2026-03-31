# Copilot Workspace Instructions

Use the repository structure knowledge base in .github as the default context map for this project.

## Rules
- Start each task by mapping the request to known subsystems from the repo map before broad searching.
- Prefer editing within the smallest subsystem boundary that satisfies the request.
- When uncertain, verify with source files and update assumptions.
- Treat notebooks as runtime entry points and libs as reusable implementation modules.
- Do not infer architecture beyond what is documented in code and the repo map.

## Priority Reading Order
1. README
2. pyproject
3. Repo structure knowledge base markdown in .github
4. Relevant subsystem files in libs
5. Notebook entry points only when task is workflow-facing

## Maintenance
- When architecture changes, update the repo structure knowledge base in .github in the same change.
- Keep descriptions concrete, short, and evidence-based.