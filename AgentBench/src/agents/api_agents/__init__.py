try:
    from .openai_agents import OpenAIChatCompletion, OpenAICompletion
except:
    print("> [Warning] OpenAI Agents are not available")
try:
    from .llm_assassin import OpenAIChatCompletionAssassin
except:
    print("> [Warning] Assassin Agents are not available")
try:
    from .avalon_claude import AvalonClaude
except:
    print("> [Warning] Avalon Claude is not available")
try:
    from .claude_agents import Claude
except:
    print("> [Warning] Claude Agents are not available")
