# AI Orchestrator Plugin Development Guide

## Introduction

The AI Orchestrator plugin system allows extending program functionality without modifying the core code. Plugins are Python modules that can add new actions, process messages, and modify system behavior.

## Plugin Structure

Each plugin must:
1. Be a Python file (.py) in the `plugins/` directory
2. Contain a class inheriting from `BasePlugin`
3. Implement required methods

## Basic Plugin Structure

```python
from plugins.base_plugin import BasePlugin
from typing import Dict, Any, List

class MyPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "MyPlugin"
        self.version = "1.0.0"
        self.description = "My first plugin"
        self.author = "Your Name"
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "actions": self.get_available_actions()
        }
    
    def get_available_actions(self) -> List[str]:
        return ["my_action"]
    
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        if action == "my_action":
            return self.handle_my_action(data, orchestrator)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def handle_my_action(self, data: Dict[str, Any], orchestrator) -> str:
        # Your logic here
        return "Action executed!"
```

## Required Methods

### `get_plugin_info(self) -> Dict[str, Any]`
Returns plugin information.

**Returns:**
- `name`: Plugin name
- `version`: Plugin version
- `description`: Functionality description
- `author`: Plugin author
- `actions`: List of available actions

### `get_available_actions(self) -> List[str]`
Returns list of action names that the plugin can execute.

### `execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any`
Executes specified plugin action.

**Parameters:**
- `action`: Action name
- `data`: Action data (from JSON request)
- `orchestrator`: Reference to main orchestrator class

## Optional Methods (Hooks)

### `initialize(self, orchestrator) -> bool`
Called when plugin is loaded. Use for resource initialization.

### `cleanup(self) -> None`
Called when plugin is unloaded. Use for resource cleanup.

### `on_message_received(self, message: str, orchestrator) -> Optional[str]`
Called when message is received from user (before main processing).
Can modify message or return `None` to continue normal processing.

### `on_response_generated(self, response: str, orchestrator) -> Optional[str]`
Called after AI response generation.
Can modify response or return `None` to keep original response.

## Accessing Orchestrator Functionality

Through `orchestrator` parameter you can access:
- `orchestrator.call_brain_model(text)` - Call AI model
- `orchestrator.analyze_image_with_vision(path)` - Image analysis
- `orchestrator.extract_text_from_image(path)` - OCR
- `orchestrator.telegram_bot` - Telegram bot (if enabled)
- `orchestrator.logger` - Logger
- And other main class methods

## Plugin Configuration

Plugins are configured through `plugins/config.ini`:

```ini
[plugins]
my_plugin=true
another_plugin=false
```

- `true` - plugin enabled
- `false` - plugin disabled

## Using Plugins in JSON Actions

To call plugin action use format:

```json
{
  "action": "plugin:plugin_name:action_name",
  "data": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

Where:
- `plugin_name` - plugin name (.py filename)
- `action_name` - action name from `get_available_actions()`

## Logging

Each plugin has its own logger:

```python
self.logger.info("Information message")
self.logger.warning("Warning")
self.logger.error("Error")
```

## Error Handling

Use `PluginError` for plugin-specific errors:

```python
from plugins.base_plugin import PluginError

if not data.get("required_param"):
    raise PluginError("Missing required parameter")
```

## Usage Examples

### Simple Calculator Plugin

```python
from plugins.base_plugin import BasePlugin
from typing import Dict, Any, List

class CalculatorPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "Calculator"
        self.version = "1.0.0"
        self.description = "Simple calculator"
        self.author = "AI Orchestrator"
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "actions": self.get_available_actions()
        }
    
    def get_available_actions(self) -> List[str]:
        return ["calculate"]
    
    def execute_action(self, action: str, data: Dict[str, Any], orchestrator) -> Any:
        if action == "calculate":
            expression = data.get("expression", "")
            try:
                result = eval(expression)  # In production use safe parser
                return f"Result: {expression} = {result}"
            except Exception as e:
                return f"Calculation error: {e}"
        else:
            raise ValueError(f"Unknown action: {action}")
```

### JSON for calling:

```json
{
  "action": "plugin:calculator:calculate",
  "expression": "2 + 2 * 3"
}
```

## Best Practices

1. **Naming**: Use clear names for plugins and actions
2. **Documentation**: Add docstrings to methods
3. **Error Handling**: Always handle possible exceptions
4. **Logging**: Use logs for debugging and monitoring
5. **Resources**: Release resources in `cleanup()` method
6. **Performance**: Avoid heavy operations in `initialize()`

## Plugin Debugging

1. Check console logs when plugin loads
2. Use `self.logger` for debug messages
3. Ensure plugin class inherits from `BasePlugin`
4. Check Python syntax in plugin file
5. Ensure plugin is enabled in `config.ini`

## Limitations

1. Plugin filename must be valid Python identifier
2. Only one plugin class per file
3. Plugins cannot directly modify other plugins' configuration
4. Avoid action name conflicts between plugins
