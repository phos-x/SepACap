import inspect
from typing import Callable, Dict, Any, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: List[Dict[str, Any]] = []

    def register(self, name: str, description: str):
        """Decorator to register a tool so the LLM can see it."""
        def decorator(func: Callable):
            self._tools[name] = func
            
            # Auto-extract the arguments the function needs
            sig = inspect.signature(func)
            params = {
                p_name: str(p.annotation.__name__) if hasattr(p.annotation, '__name__') else "Any"
                for p_name, p in sig.parameters.items() 
                if p_name not in ['optimizer', 'multi_loss', 'model'] # Hide PyTorch objects from LLM args
            }
            
            self._tool_descriptions.append({
                "tool_name": name,
                "description": description,
                "required_args": params
            })
            return func
        return decorator

    def get_tool_prompt(self) -> str:
        """Generates the text to inject into the LLM's prompt."""
        import json
        return json.dumps(self._tool_descriptions, indent=2)

    def execute(self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Executes the requested tool, injecting PyTorch context natively."""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found."
        
        func = self._tools[tool_name]
        
        # Inject the heavy PyTorch objects silently so the LLM doesn't have to worry about them
        for key in ['optimizer', 'multi_loss', 'model']:
            if key in inspect.signature(func).parameters and key in context:
                kwargs[key] = context[key]
                
        try:
            func(**kwargs)
            return f"Successfully executed {tool_name}."
        except Exception as e:
            return f"Execution failed for {tool_name}: {e}"

# Global registry instance
registry = ToolRegistry()