from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatAnthropic
import warnings
import os
from abc import ABC, abstractmethod
import tiktoken
from strategist.searchlight.utils import AbstractLogged

class LLMModel(AbstractLogged):
    '''
    Abstract class for LLM models
    '''
    def __init__(self) -> None:
        self.num_calls = 0
        self.num_generated_responses = 0
        super().__init__()

    def generate(self, input_prompt: str, num_responses=1, temperature=0.7) -> list[str]:
        '''
        Generate a response to an input prompt

        Args:
            input_prompt: input prompt
            num_responses: number of responses to generate
            temperature: temperature to use for generation

        Returns:
            list of generated responses
        '''
        self.num_calls += 1
        self.num_generated_responses += num_responses
        self.logger.info(f"Generating response to input prompt: \n {input_prompt}")
        output = self._generate(input_prompt, num_responses, temperature)
        self.logger.info(f"Generated response: \n {output[0]}")
        return output
    
    @abstractmethod
    def get_num_total_tokens(self) -> int:
        pass
    
    @abstractmethod
    def get_num_output_tokens(self) -> int:
        pass

    def get_num_calls(self) -> int:
        return self.num_calls

    @abstractmethod
    def _generate(self, input_prompt: str, num_responses=1, temperature: float=0.7) -> list[str]:
        '''
        Generate a response to an input prompt

        Args:
            input_prompt: input prompt
            num_responses: number of responses to generate
            temperature: temperature to use for generation

        Returns:
            list of generated responses
        '''
        pass
    
class GPT35Multi(LLMModel):
    def __init__(self, temperature: float=0.7, num_responses=1, api_key=None, model="gpt-3.5-turbo", max_expense: float=40):
        super().__init__()
        
        if api_key is not None:
            self.key = api_key
        else:
            self.key = os.environ.get("OPENAI_API_KEY")
        # print(num_responses)
        self.num_responses = num_responses
        self.temperature = temperature

        # record expense
        self.total_expense = 0
        self.MAX_EXPENSE = max_expense

        # record num_input_tokens
        self.num_input_tokens = 0

        # record num_output_tokens
        self.num_output_tokens = 0

        self.enc = tiktoken.encoding_for_model(model)

        # Initialize the model with the desired temperature and number of responses
        self.model = ChatOpenAI(temperature=temperature, model=model, openai_api_key=self.key, n=num_responses)

        if model == "gpt-3.5-turbo":
            self.cost_per_input_token = 5e-7
            self.cost_per_output_token = 1.5e-6
        elif model == "gpt-4":
            self.cost_per_input_token = 3e-5
            self.cost_per_output_token = 6e-5
        elif model == "gpt-4-turbo":
            self.cost_per_input_token = 1e-5
            self.cost_per_output_token = 3e-5

    def get_num_total_tokens(self) -> int:
        return self.num_input_tokens + self.num_output_tokens
    
    def get_num_output_tokens(self) -> int:
        return self.num_output_tokens

    def single_action(self, input_prompt: str, num_responses=-1, temperature: float=0.7) -> list[str]:
        warnings.warn("single_action() will be deprecated, please use generate() instead")
        if num_responses == -1:
            num_responses = self.num_responses
        
        # Initialize the model with the desired temperature and number of responses
        self.model = ChatOpenAI(temperature=temperature, openai_api_key=self.key, n=num_responses)

        # Wrap the input prompt in a list of HumanMessage objects as before
        input_prompt = [HumanMessage(content=input_prompt)]
        
        # Invoke the model and get the output
        outputs = self.model._generate(input_prompt)
        
        # Assuming outputs is a list of responses, return it directly
        return [generation.text for generation in outputs.generations]
    
    def _generate(self, input_prompt: str, num_responses=-1, temperature: float=-1.0) -> list[str]:
        num_input_tokens = len(self.enc.encode(input_prompt))
        self.total_expense += num_input_tokens * 5e-7
        self.num_input_tokens += num_input_tokens
        if self.total_expense > self.MAX_EXPENSE:
            raise RuntimeError(f"Exceed max expense. Currect expense: ${self.total_expense}. Total calls: {self.num_calls}. Total responses: {self.num_generated_responses}")
        if num_responses != -1:
            temp_num_responses = self.model.n
            self.model.n = num_responses

        if temperature != -1.0:
            temp_temperature = self.model.temperature
            self.model.temperature = temperature

        # Wrap the input prompt in a list of HumanMessage objects as before
        input_prompt = [HumanMessage(content=input_prompt)]
        
        # Invoke the model and get the output
        outputs = self.model._generate(input_prompt)

        if num_responses != -1:
            self.model.n = temp_num_responses

        if temperature != -1.0:
            self.model.temperature = temp_temperature
        
        num_output_tokens = sum([len(self.enc.encode(generation.text)) for generation in outputs.generations])  
        self.num_output_tokens += num_output_tokens      
        self.total_expense += num_output_tokens * 1.5e-6
        
        if self.total_expense > self.MAX_EXPENSE:
            raise RuntimeError(f"Exceed max expense. Currect expense: ${self.total_expense}. Total calls: {self.num_calls}. Total responses: {self.num_generated_responses}")
        # Assuming outputs is a list of responses, return it directly
        return [generation.text for generation in outputs.generations]
    
class ClaudeMulti(LLMModel):
    def __init__(self, temperature: float=0.7, num_responses=1, api_key=None, model="claude-2", max_expense: float=20):
        super().__init__()
        
        if api_key is not None:
            self.key = api_key
        else:
            self.key = os.environ.get("CLAUDE_API_KEY")
        # print(num_responses)
        self.num_responses = num_responses
        self.temperature = temperature

        # record expense
        self.total_expense = 0
        self.MAX_EXPENSE = max_expense

        self.enc = tiktoken.get_encoding("cl100k_base")

        # Initialize the model with the desired temperature and number of responses
        self.model = ChatAnthropic(temperature=temperature, model=model, anthropic_api_key=self.key)
    
    def _generate(self, input_prompt: str, num_responses=-1, temperature: float=-1.0) -> list[str]:
        num_input_tokens = len(self.enc.encode(input_prompt))
        self.total_expense += num_input_tokens * 3e-5
        if self.total_expense > self.MAX_EXPENSE:
            raise RuntimeError(f"Exceed max expense. Currect expense: ${self.total_expense}")
        n = 1
        if num_responses != -1:
            # temp_num_responses = self.model.n
            n = num_responses

        if temperature != -1.0:
            temp_temperature = self.model.temperature
            self.model.temperature = temperature

        # Wrap the input prompt in a list of HumanMessage objects as before
        input_prompt = [HumanMessage(content=input_prompt)]
        
        # Invoke the model and get the output
        # outputs = []
        # for _ in range(n):
        outputs = self.model._generate(input_prompt)
        #     outputs.append(output.generations.text)
        outputs = [generation.text for generation in outputs.generations]

        # if num_responses != -1:
        #     self.model.n = temp_num_responses

        if temperature != -1.0:
            self.model.temperature = temp_temperature
        
        num_output_tokens = sum([len(self.enc.encode(generation)) for generation in outputs])        
        self.total_expense += num_output_tokens * 6e-5
        if self.total_expense > self.MAX_EXPENSE:
            raise RuntimeError(f"Exceed max expense. Currect expense: ${self.total_expense}")
        # Assuming outputs is a list of responses, return it directly
        return outputs
    
    def get_num_total_tokens(self) -> int:
        return 0
    
    def get_num_output_tokens(self) -> int:
        return 0