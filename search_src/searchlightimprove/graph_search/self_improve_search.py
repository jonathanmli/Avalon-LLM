from search_src.searchlightimprove.headers import *
from search_src.searchlight.headers import *
from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from search_src.searchlight.datastructures.graphs import *

# TODO: deprecated

class SelfImprovementInitialInferencer(InitialInferencer2):
    def __init__(self, improvement_proposer: ImprovementProposer, evaluator: Evaluator, graph: ValueGraph2):
        super().__init__()
        self.improvement_proposer = improvement_proposer
        self.evaluator = evaluator
        self.graph = graph # NOTE: this is a temporary fix, we should not need to pass the graph

    def predict(self, state: State) -> tuple[dict, dict, dict[tuple[tuple[Any, Any],...],Any], dict[tuple[tuple[Any, Any],...],Any], dict]:
        # if terminal state, return empty sets of everything
        if state.notes['done']:
            return dict(), dict(), dict(), dict(), dict()

        actors = {0}

        # get the heuristics feedback from the notes of the node
        node = self.graph.get_node(state)
        if node is not None:
            notes = node.notes['heuristic_feedback']
        else:
            notes = dict()

        print('notes:', notes)

        # propose improved functions
        # TODO: we need to somehow pass the notes to the improvement proposer
        improved_functions = self.improvement_proposer.propose(state.id, notes)

        # evaluate the improved functions
        scores, feedback_notes = self.evaluator.evaluate(improved_functions)

        # the rest of the code formats the outputs into the correct format

        # create joint actions
        non_joint_actions = list(range(len(improved_functions)+1))
        joint_actions = set([((0, action),) for action in non_joint_actions])

        # create next states
        next_states = set()
        # create a state for each improved function
        for i, improved_function in enumerate(improved_functions):
            next_state = State(improved_function)
            next_state.notes = {'score': scores[i], 'done': False}
            # append notes[i] to next_state.notes
            for key, value in notes[i].items():
                next_state.notes[key] = value
            next_states.add(next_state)
        

        # create a terminal state corresponding to ending at current state
        terminal_state = State(state.id + '_terminal')
        terminal_state.notes = {'score': state.notes['score'], 'done': True} # TODO: check if correct
        next_states.add(terminal_state)
        scores.append(state.notes['score'])

        # create policies, which should be uniform random
        actor_to_action_to_prob = {0: {action: 1/(len(improved_functions)+1) for action in non_joint_actions}}


        # create next state values
        next_state_values = {next_state: {0: score} for next_state, score in zip(next_states, scores)}
        # create intermediate rewards
        action_to_actor_to_reward = {action: {0: 0} for action in joint_actions}
        # change reward of terminal state to current state's score
        action_to_actor_to_reward[((0, len(improved_functions)),)] = {0: state.notes['score']}
        # create next state to next state notes
        next_state_to_notes = {next_state: notes for next_state, notes in zip(next_states, feedback_notes)}

        # create action to next state
        action_to_next_state = {action: next_state for action, next_state in zip(joint_actions, next_states)}

        return actor_to_action_to_prob, next_state_values, action_to_actor_to_reward, action_to_next_state, {'next_state_to_heuristic_notes': next_state_to_notes}