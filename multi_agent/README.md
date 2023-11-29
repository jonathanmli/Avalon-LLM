# Multi-agent Submodule

**Usage**: Instantiate the `MultiAgentProxy` class with the task session and the number of agents. Call `set_current_agent` to set the current agent that will take the control of the task session. For advanced and more complex multi-agent behaviours, e.g. cooperation, you can override `get_next_agent`, and call this method to switch between agents.