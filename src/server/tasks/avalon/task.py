import sys
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Any

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, TaskOutput, SampleIndex, AgentOutputStatus, SampleStatus
from src.utils import ColorMessage

from .engine import *
from .task_scoring import *

from .prompts import *
from .agents.baseline_agents import *
from .agents.baseline_speakable_agents import *

from .wrapper import FakeSession, AvalonSessionWrapper
from .utils import verbalize_team_result, verbalize_mission_result

from .agents.llm_with_discussion import LLMAgentWithDiscussion
from .agents.search_agent import SearchlightLLMAgentWithDiscussion

from src.typings import AgentContextLimitException
from .avalon_exception import AvalonAgentActionException

from multi_agent.proxy import MultiAgentProxy
import logging
from strategist.dialogue_improve.data_loader import DataLoader
from strategist.Avalon.baseline_models_Avalon import AvalonState
from good_examples.Avalon.value_heuristics.list import functions as avalon_func
from .dialogue import AvalonDiagloue
from good_examples.Avalon.english import role_to_guide as search_guide
from good_examples.Avalon.recon import role_to_guide as recon_guide

AGENT_FINDER = {
    'naive': find_naive_agent,
    'speak_naive': find_naive_speakable_agent,
    'llm': LLMAgentWithDiscussion,
    'search': SearchlightLLMAgentWithDiscussion,
    'recon': SearchlightLLMAgentWithDiscussion
}
dialogue_guides = {
    'recon': recon_guide,
    'search': search_guide
}

class AvalonBench(Task):
    def __init__(self, num_players, agent_list, discussion, data_file, **configs):
        super().__init__(**configs)

        self.num_players = num_players
        self.agent_list = agent_list

        self.discussion = discussion
        self.data_file = data_file


        self.data: List[Tuple[dict, set]] = []
        with open(self.data_file, "r") as f:
            data_object = json.load(f)
        for data_item in data_object:
            self.data.append((data_item, -1))
        self.inputs = data_object

        self.seed = configs.pop('seed', 0)
        self.FILE_PATH = 'search_src/dialogue_improve/new_data04.jsonl'
        self.discussion_history = []
        self.data_loader = DataLoader()

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        outputs = [None for _ in range(len(self.data))]
        for result in results:
            outputs[result.index] = result.result

        win_counter = 0
        deduc_acc = 0
        valid_games = 0
        for result in results:
            if result.status == SampleStatus.COMPLETED:
                llm_idx = result.result['llm_idx']
                if result.result[f'Player_{llm_idx}_wins']:
                    win_counter += 1
                deduc_acc += result.result[f'Player_{llm_idx}_deduc_acc']
                valid_games += 1
        

        return {
            "Win rate of Player 0": win_counter / len(outputs),
            "Avg deduction acc of Player 0": deduc_acc / len(outputs),
        }

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        assert isinstance(index, int), "Index must be an integer"
        assert self.inputs[index]['num_players'] == self.num_players, "Number of players must be the same"
        proxy = MultiAgentProxy(session, self.num_players)
        sessions = [AvalonSessionWrapper(session, proxy) for _ in range(self.num_players)]
        proxy.initialize_sessions(sessions)
        env = AvalonGameEnvironment.from_presets(self.inputs[index])
        scoring = AvalonScoring(env.config)

        true_player_sides = []
        believed_player_sides = []
        game_env_log = []

        llm_idx = 0

        num_players = self.num_players

        player_list = []

        if num_players != len(sessions):
            raise ValueError(
                f"Number of players {num_players} doesn't match number of sessions {len(sessions)}"
            )
        
        dialogue_history = AvalonDiagloue()

        print("Check initialization")
        # Initialize players. Please remember to let Merlin and Evil players see the sides of all players.
        for i, (role_i, role_name, side) in enumerate(env.get_roles()):
            if self.agent_list[i] == 'search' or self.agent_list[i] == 'recon':
                player_list.append(AGENT_FINDER[self.agent_list[i]](
                                        id          =   i,
                                        name        =   f"Player {i}",
                                        config      =   env.config,
                                        side        =   side,
                                        role        =   role_i,
                                        num_players =   num_players,
                                        func_str    =   avalon_func[1],
                                        session     =   sessions[i],
                                        role_name   =   role_name,
                                        merlin      =   env.config.merlin,
                                        percival    =   env.config.percival,
                                        morgana     =   env.config.morgana,
                                        mordred     =   env.config.mordred,
                                        oberon      =   env.config.oberon,
                                        num_good    =   env.config.num_good,
                                        num_evil    =   env.config.num_evil,
                                        discussion  =   self.discussion,
                                        seed        =   self.seed,
                                        guides      =   dialogue_guides[self.agent_list[i]] # TODO: seed
                                        ))
            else:
                player_list.append(AGENT_FINDER[self.agent_list[i]](
                                        id          =   i,
                                        name        =   f"Player {i}",
                                        config      =   env.config,
                                        side        =   side,
                                        role        =   role_i,
                                        num_players =   num_players,
                                        session     =   sessions[i],
                                        role_name   =   role_name,
                                        merlin      =   env.config.merlin,
                                        percival    =   env.config.percival,
                                        morgana     =   env.config.morgana,
                                        mordred     =   env.config.mordred,
                                        oberon      =   env.config.oberon,
                                        num_good    =   env.config.num_good,
                                        num_evil    =   env.config.num_evil,
                                        discussion  =   self.discussion,
                                        seed        =   self.seed # TODO: seed
                                        ))
            # If the player is Merlin or Evil, let them see the sides of all players.
            player_sides = [side for _, _, side in env.get_roles()]
            if player_list[i].role == 0 or player_list[i].side == 0:
                player_list[i].see_sides(player_sides)
                await player_list[i].initialize_game_info(player_list=env.get_roles(), env=env)
            else:
                await player_list[i].initialize_game_info(player_list=env.get_roles(), env=env)
            
            proxy.get_next_agent()
        
        # try:
        while not env.done:
            phase = env.get_phase()[0]
            print()
            print(ColorMessage.orange(f"##### Mission {env.turn}, Round {env.round} #####"))
            
            # if phase is team selection phase, ask for team
            if phase == 0:
                leader = env.get_quest_leader()
                game_env_log.append(f"Selection Phase, the leader is Player {leader}")
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print(f"Selection Phase, the leader is Player {leader}")
                """
                Leader speaks & Discussion
                """
                speaking_order = []
                private_informations = []
                roles = []
                # intended_team_list = []
                if self.discussion:
                    print()
                    print(ColorMessage.cyan(f"##### Discussion Starts #####"))
                    print()
                    # dialogue_history: list[tuple[int, str]] = []
                    # Leader speaks
                    summaries = []
                    proxy.set_current_agent(leader)
                    for idx, player in enumerate(player_list):
                        proxy.set_current_agent(idx)
                        summary_item = await player.summarize(env=env)
                        summaries.append(str(summary_item[0].content))
                    # print("Test: ", player_list[leader].team_discussion)
                    # team, statement = await player_list[leader].test()
                    # print(leader)
                    # print(player_list[leader].team_discussion)
                    proxy.set_current_agent(leader)
                    dialogue = await player_list[leader].team_discussion(
                            team_size           =   env.get_team_size(),
                            team_leader_id      =   leader,
                            mission_id          =   env.turn,
                            env                 =   env,
                            dialogue_history    =   dialogue_history,
                        )
                    print(ColorMessage.blue(f"Player {leader}(Leader):") + " " + dialogue)
                    roles.append(player_list[leader].role)
                    dialogue_history.append(leader, dialogue)
                    self.discussion_history.append(f"Player {leader}:\n{dialogue}")
                    speaking_order.append(leader)
                    private_informations.append(player_list[leader].system_info)
                    # intended_team = await player_list[leader].propose_team(
                    #     team_size           =   env.get_team_size(),
                    #     mission_id          =   env.turn,
                    #     env                 =   env,
                    # )
                    # intended_team_list.append(list(intended_team))

                    # Discussion (sequential, once, in order for now) and Summarize
                    for idx in range(leader+1, leader + num_players):
                        player_id = idx % num_players
                        player = player_list[player_id]
                    # for idx, player in enumerate(player_list):
                        proxy.set_current_agent(player_id)
                        dialogue = await player.team_discussion(
                            team_size           =   env.get_team_size(),
                            team_leader_id      =   leader,
                            mission_id          =   env.turn,
                            dialogue_history    =   dialogue_history,
                            env                 =   env,
                        )
                        print(ColorMessage.blue(f"Player {player_id}:") + " " + dialogue)
                        roles.append(player.role)
                        dialogue_history.append(player_id, dialogue)
                        self.discussion_history.append(f"Player {player_id}:\n{dialogue}")
                        speaking_order.append(player_id)
                        private_informations.append(player.system_info)

                    # query the intended teams after discussion
                    # for idx, player in enumerate(player_list):
                    #     proxy.set_current_agent(idx)
                    #     if idx == leader:
                    #         continue
                    #     intended_team = await player.propose_team(
                    #         team_size           =   env.get_team_size(),
                    #         mission_id          =   env.turn,
                    #         env                 =   env,
                    #     )
                    #     intended_team_list.append(list(intended_team))
                    # for idx, player in enumerate(player):
                    #     proxy.set_current_agent(idx)
                    #     player.discussion_end(
                    #         leader              =   leader,
                    #         leader_statement    =   statement,
                    #     )
                    # for idx, player in enumerate(player_list):
                    
                    # print(AvalonState.init_from_env(env).get_state_tuple())
                    # print(dialogue_history.dialogue_tuple_to_list())
                    # self.data_loader.add_data_point(
                    #     discussion_history_summary=summaries,
                    #     state_info=AvalonState.init_from_env(env).get_state_tuple(),
                    #     intended_actions=intended_team_list,
                    #     private_informations=private_informations,
                    #     roles=roles,
                    #     dialogue=dialogue_history.dialogue_tuple_to_list(),
                    #     speaking_order=speaking_order,
                    # )
                    # self.data_loader.save_data(self.FILE_PATH)
                # Choose a team
                # print(player_list[leader].propose_team)
                    print(ColorMessage.cyan(f"##### Discussion Ends #####"))
                proxy.set_current_agent(leader)
                team = await player_list[leader].propose_team(
                    team_size           =   env.get_team_size(),
                    mission_id          =   env.turn,
                    env                 =   env,
                )
                env.choose_quest_team(
                    team   =  frozenset(team),
                    leader =  leader
                )
                game_env_log.append(f"Leader Player {leader} chooses team {list(team)}")
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print(f"Leader Player {leader} chooses team {list(team)}")

            # if phase is team voting phase, ask for votes
            elif phase == 1:
                game_env_log.append("Team Voting Phase")
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print("Team voting Phase")
                votes = []
                proxy.set_current_agent(0)
                for i in range(num_players):
                    proxy.set_current_agent(i)
                    vote = await player_list[i].vote_on_team(
                        team                =   env.get_current_quest_team(),
                        mission_id          =   env.turn,
                        env                 =   env,
                        )
                    votes.append(vote)
                # votes = [
                #     await player_list[i].vote_on_team(
                #         team                =   env.get_current_quest_team(),
                #         mission_id          =   env.turn,
                #         env                 =   env,
                #         ) for i in range(num_players)
                #         ]
                try:
                    outcome = env.gather_team_votes(votes)
                except Exception as e:
                    print(e)
                game_env_log.append(f"Team votes at this round: {str(votes)}")

                # Observe results of Team Selection
                for idx, player in enumerate(player_list):
                    proxy.set_current_agent(idx)
                    await player.observe_team_result(
                        mission_id  =   env.turn,
                        team        =   env.get_current_quest_team(),
                        votes       =   votes,
                        outcome     =   outcome[2],
                    )

                game_env_log.append("Team result: " + verbalize_team_result(team=env.get_current_quest_team(), votes=votes, outcome=outcome[2]))
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print("Team result: " + verbalize_team_result(team=env.get_current_quest_team(), votes=votes, outcome=outcome[2]))


            # if phase is quest voting phase, ask for votes
            elif phase == 2:
                game_env_log.append("Quest Voting Phase")
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print("Quest Voting Phase")
                '''
                TODO: Can have a discussion before voting on quest
                '''
                votes = []
                for i in env.get_current_quest_team():
                    proxy.set_current_agent(i)
                    vote = await player_list[i].vote_on_mission(
                        team                =   env.get_current_quest_team(),
                        mission_id          =   env.turn,
                        env                 =   env,
                        )
                    votes.append(vote)
                votes = [
                    await player_list[i].vote_on_mission(
                        team                =   env.get_current_quest_team(),
                        mission_id          =   env.turn,
                        env                 =   env,
                        ) for i in env.get_current_quest_team()
                        ]
                outcome = env.gather_quest_votes(votes)
                game_env_log.append(f"Quest votes at this round: {str(votes)}")

                # Observe mission/quest result
                proxy.set_current_agent(0)
                for idx, player in enumerate(player_list):
                    proxy.set_current_agent(idx)
                    await player.observe_mission(
                        team        =   env.get_current_quest_team(),
                        mission_id  =   env.turn-1,
                        num_fails   =   outcome[3],
                        votes       =   votes,
                        outcome     =   outcome[2],
                        env         =   env,
                    )

                game_env_log.append("Quest result: " + verbalize_mission_result(team=env.get_current_quest_team(), outcome=outcome[2]))
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print("Quest result: " + verbalize_mission_result(team=env.get_current_quest_team(), outcome=outcome[2]))
            
            # if phase is assassination phase, ask for assassination
            elif phase == 3:
                game_env_log.append("Assassination phase")
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print("Assassination phase")
                '''
                    TODO: Discussion before Assassination Phase
                '''
                # assassin = env.get_assassin()
                for idx, player in enumerate(player_list):
                    if player.role == 7:
                        assassin = idx
                proxy.set_current_agent(assassin)
                target = int(await player_list[assassin].assassinate(
                    env=env,
                    ))
                _, _, assassinated = env.choose_assassination_target(assassin, target)
                game_env_log.append(f"Assassin Player {assassin} chooses to assassinate Player {target}")
                print()
                print(ColorMessage.cyan(f"##### System #####"))
                print()
                print(f"Assassin Player {assassin} chooses to assassinate Player {target}")
        # reflect sides of each player at the end of the game
        for idx, player in enumerate(player_list):
            proxy.set_current_agent(idx)
            if idx == llm_idx:
                llm_believed_player_sides = await player.get_believed_sides(
                    num_players =   self.num_players,
                    env         =   env,
                    )

                true_player_sides.append(list(map(int, env.is_good)))
                believed_player_sides.append(llm_believed_player_sides)

        if env.good_victory:
            answer = 1
        else:
            if sum(env.quest_results) >= 3:
                answer = 0
            else:
                answer = -1
        finish_reason = SampleStatus.COMPLETED

        # except AgentContextLimitException as e1:
        #     return TaskSampleExecutionResult(status=SampleStatus.AGENT_CONTEXT_LIMIT)
        # except AvalonAgentActionException as e2:
        #     return TaskSampleExecutionResult(status=SampleStatus.AGENT_INVALID_ACTION, result={"result": False, "error": e2})
        # except Exception as e:
        #     finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
        #     return TaskSampleExecutionResult(status=finish_reason, result={"result": False, "error": e})
        
        verbal_game_result = {
            -1: "Evil wins by mission!",
            0: "Evil wins by assassination!",
            1: "Good wins!"
        }
        return TaskSampleExecutionResult(status=finish_reason, result={
            "game_result": verbal_game_result[answer],
            "llm_idx": llm_idx,
            f"role_of_Player_{llm_idx}": player_list[llm_idx].role_name,
            f"Player_{llm_idx}_wins": (answer > 0) == bool(player_list[llm_idx].side),
            f"Player_{llm_idx}_deduc_acc": scoring.deduction_acc(true_player_sides, believed_player_sides),
            "game_env_log": game_env_log,
            "history for player 0": proxy.history[0],
            "history for player 1": proxy.history[1],
            "history for player 2": proxy.history[2],
            "history for player 3": proxy.history[3],
            "history for player 4": proxy.history[4],
        })