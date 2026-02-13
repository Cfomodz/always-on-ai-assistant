"""Tests for tools registry."""
from pathlib import Path

# Import after potential env setup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.tools_registry import (
    get_enabled_tools,
    extract_commands_from_source,
    filter_source_by_enabled,
)


def test_extract_commands_from_source():
    """Extract typer commands from Python source."""
    source = '''
@app.command()
def ping_server():
    """Ping."""
    pass

@app.command()
def weather(location: str):
    """Weather."""
    pass
'''
    cmds = extract_commands_from_source("test.py", source)
    assert "ping_server" in cmds
    assert "weather" in cmds
    assert "def weather" in cmds["weather"]


def test_get_enabled_tools_default():
    """Enabled tools use default when no state override."""
    config = {
        "tools": {
            "a": {"default_enabled": True},
            "b": {"default_enabled": False},
        }
    }
    state = {}
    enabled = get_enabled_tools(config, state)
    assert "a" in enabled
    assert "b" not in enabled


def test_get_enabled_tools_with_override():
    """State overrides default."""
    config = {
        "tools": {
            "a": {"default_enabled": True},
            "b": {"default_enabled": False},
        }
    }
    state = {"b": True}
    enabled = get_enabled_tools(config, state)
    assert "a" in enabled
    assert "b" in enabled


def test_filter_source_by_enabled(tmp_path):
    """Filter source to only enabled commands."""
    test_file = tmp_path / "commands.py"
    test_file.write_text('''
app = None

@app.command()
def foo():
    """Foo."""
    pass

@app.command()
def bar():
    """Bar."""
    pass
''')
    filtered = filter_source_by_enabled(
        [str(test_file)],
        {"foo"},
    )
    assert "def foo" in filtered
    assert "def bar" not in filtered
