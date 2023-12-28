from headers import ForwardEnumerator

class GPT35ForwardEnumerator(ForwardEnumerator):
    '''
    Forward dynamics enumerator for GPT-3.5
    '''

    def __init__(self, model, tokenizer, device):
        '''
        Args:
            model: GPT-3.5 model
            tokenizer: GPT-3.5 tokenizer
            device: torch device
        '''
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def enumerate(self, state, action):
        '''
        Enumerates the possible next states given the current state and action

        Args:
            state: current state
            action: action to take

        Returns:
            next_states: list of next states
        '''
        # Encode state and action
        state = self.tokenizer.encode(state)
        action = self.tokenizer.encode(action)

        # Concatenate state and action
        input_ids = state + action

        # Convert to tensor and move to device
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate next states
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, max_length=100, num_beams=10, num_return_sequences=10, temperature=1.0, repetition_penalty=1.0, do_sample=True, top_k=50, top_p=0.95, pad_token_id=0, eos_token_id=50256)

        # Decode next states
        next_states = []
        for output in outputs:
            next_state = self.tokenizer.decode(output, skip_special_tokens=True)
            next_states.append(next_state)

        return next_states