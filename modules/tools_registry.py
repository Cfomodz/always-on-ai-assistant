"""
Tools registry - manages enabled/default_enabled state for typer commands.
Only enabled tools are fed to the classifier and typer agent.
"""
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Set

import yaml

TOOLS_STATE_FILE = "tools_state.json"
TOOLS_CONFIG_FILE = "tools_config.yml"


def _state_path() -> Path:
    return Path(TOOLS_STATE_FILE)


def _config_path() -> Path:
    return Path(TOOLS_CONFIG_FILE)


def load_tools_config() -> Dict:
    """Load tools config (default_enabled values). Creates default if missing."""
    config_path = _config_path()
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_tools_state() -> Dict[str, bool]:
    """Load persisted enabled state overrides. Empty if none."""
    state_path = _state_path()
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_tools_state(state: Dict[str, bool]) -> None:
    """Persist enabled state overrides."""
    with open(_state_path(), "w") as f:
        json.dump(state, f, indent=2)


def get_enabled_tools(tools_config: Dict, tools_state: Dict) -> Set[str]:
    """Get set of tool names that are currently enabled."""
    enabled = set()
    for name, cfg in tools_config.get("tools", {}).items():
        default = cfg.get("default_enabled", True)
        if name in tools_state:
            if tools_state[name]:
                enabled.add(name)
        elif default:
            enabled.add(name)
    return enabled


def set_tool_enabled(name: str, enabled: bool, tools_config: Dict) -> bool:
    """
    Set a tool's enabled status. Returns True if tool exists and was updated.
    Persists to tools_state.json.
    """
    tools = tools_config.get("tools", {})
    if name not in tools:
        return False
    state = load_tools_state()
    state[name] = enabled
    save_tools_state(state)
    return True


def get_all_tool_names(tools_config: Dict) -> List[str]:
    """Get all registered tool names."""
    return list(tools_config.get("tools", {}).keys())


def is_tool_enabled(name: str, tools_config: Dict, tools_state: Dict) -> bool:
    """Check if a specific tool is enabled."""
    if name not in tools_config.get("tools", {}):
        return False
    if name in tools_state:
        return tools_state[name]
    return tools_config["tools"][name].get("default_enabled", True)


def extract_commands_from_source(file_path: str, source: str) -> Dict[str, str]:
    """
    Extract typer command definitions from Python source.
    Returns dict mapping command_name (Python name with underscore) -> source block.
    """
    result = {}
    try:
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.decorator_list:
                    continue
                for dec in node.decorator_list:
                    # Check for @app.command() or similar
                    if isinstance(dec, ast.Call):
                        if isinstance(dec.func, ast.Attribute):
                            if dec.func.attr == "command":
                                start = node.lineno - 1
                                end = node.end_lineno  # 1-based, inclusive
                                block = "\n".join(lines[start:end])
                                result[node.name] = block
                                break
    except ast.SyntaxError:
        pass
    return result


def filter_source_by_enabled(
    typer_files: List[str], enabled_names: Set[str]
) -> str:
    """
    Load typer files, extract commands, and return concatenated source
    for enabled commands only.
    """
    out_parts = []
    for tf in typer_files:
        if not os.path.exists(tf):
            continue
        with open(tf, "r") as f:
            source = f.read()
        commands = extract_commands_from_source(tf, source)
        for name, block in commands.items():
            if name in enabled_names:
                out_parts.append(f"# --- {tf} (command: {name}) ---\n{block}\n")
    return "\n".join(out_parts) if out_parts else ""


def get_tools_registry() -> tuple[Dict, Dict, Set[str]]:
    """Load config, state, and return (config, state, enabled_set)."""
    config = load_tools_config()
    state = load_tools_state()
    enabled = get_enabled_tools(config, state)
    return config, state, enabled
