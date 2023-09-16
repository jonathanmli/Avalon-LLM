from typing import Dict
import openai
import time
import os


# TODO: can we wrap all kinds of api in a single function
def openai_wrapper(messages, temperature, **kwargs):
    executed = False
    while not executed:
        try:
            result = openai.ChatCompletion.create(
                            messages=messages,
                            temperature=temperature,
                            **kwargs
            )
            executed = True
        except Exception as e:
            print(e)
            print("Sleep for 5 seconds zzZ")
            time.sleep(5)

    return result

def load_avalon_log(game_log: Dict):
    pass

def get_statement(last_history: str):
    return last_history.split("Statement: ")[-1]

if __name__ == "__main__":
    class OpenAIChatCompletionAssassin(Agent):
        def __init__(self, api_args=None, **config):
            self.name = config.pop("name")
            if not api_args:
                api_args = {}
            print("api_args={}".format(api_args))
            print("config={}".format(config))
            
            api_args = deepcopy(api_args)
            api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
            api_args["model"] = api_args.pop("model", None)
            api_args["api_key"] = api_key
            if not api_key:
                raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
            os.environ['OPENAI_API_KEY'] = api_key
            print("OpenAI API key={}".format(openai.api_key))
            api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
            os.environ['OPENAI_API_BASE'] = api_base
            print("openai.api_base={}".format(openai.api_base))
            if not api_args["model"]:
                raise ValueError("OpenAI model is required, please assign api_args.model.")
            self.api_args = api_args
            super().__init__(**config)

        def test(self, messages, temperature, **kwargs):
            return openai_wrapper(messages, temperature, **kwargs)
    
    agent = OpenAIChatCompletionAssassin
    agent.test("hello world", 0, test="test", test2="test")
    # load_avalon_log()
        