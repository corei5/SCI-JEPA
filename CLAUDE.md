@AGENTS.md

## Claude Code

- The graphify PreToolUse hook (`.claude/settings.json`) reminds you to query the graph before raw searches. The `graphify` MCP server (`.mcp.json`) exposes the same graph as tools.
- `/graphify` loads the project skill in `.claude/skills/graphify/`.
- After code edits, run `graphify update .`. For fresh community names and views, run `bash scripts/graph-refresh.sh`.
