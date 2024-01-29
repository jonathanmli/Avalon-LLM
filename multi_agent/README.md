# Multi-agent Submodule

This module is comprised of two core parts, i.e., the `SessionWrapper` and the `MultiAgentProxy`. The `SessionWrapper` is used to provide flexible control over sessions by wrapping a `Session` class in each `SessionWrapper`. And the multi-agent feature is implemented mainly in the `MultiAgentProxy` class, and come into play by being embedded in a `SessionWrapper`. In other words, the `MultiAgentProxy` is the central controller in `SessionWrapper`.

**Usage**: Instantiate the `MultiAgentProxy` class with the task session and the number of agents. Call `set_current_agent` to set the current agent that will take the control of the task session. For advanced and more complex multi-agent behaviours, e.g. cooperation, you can override `get_next_agent`, and call this method to switch between agents.

## Demos

We also demonstrate two use cases in `demo/`, which includes *consistent conversation* and *group chat*.