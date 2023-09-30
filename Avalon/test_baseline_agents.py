from baseline_agents import *
import unittest
from engine import AvalonConfig
import numpy as np

# Test the baseline agents. Use python unittest module.
class TestNaiveServant(unittest.TestCase):
    
    # create Naive Servant
    def setUp(self):
        config = AvalonConfig(5)
        self.naive_servant = NaiveServant(3, 'Naive Servant', config)

    # test that the Naive Servant proposes a team of size 2
    def test_propose_team(self):
        self.assertEqual(len(self.naive_servant.propose_team(2)), 2)

    # test that the Naive Servant proposes a team that includes himself
    def test_propose_team_includes_self(self):
        self.assertIn(3, self.naive_servant.propose_team(2))

    # test that the Naive servant rejects any team that does not include himself when using vote_on_team
    def test_vote_on_team(self):
        self.assertEqual(self.naive_servant.vote_on_team(0, [0, 1]), 0)

    # test that the Naive servant observes team [0,1] fail
    def test_observe_team_fail(self):
        print('first test')
        self.naive_servant.observe_mission([0, 1], 0, 2)
    
    # test that the Naive servant rejects any team that is superset of the team that failed when using vote_on_team
    def test_vote_on_team_superset(self):
        self.naive_servant.observe_mission([0, 1], 0, 2)
        self.assertEqual(self.naive_servant.vote_on_team(1, [0, 1, 3]), 0)

    # test that the Naive servant only proposes a team of only good players after observing a team fail, ie. the team [2,3,4]
    def test_propose_team_after_fail(self):
        self.naive_servant.observe_mission([0, 1], 0, 2)
        self.assertEqual(self.naive_servant.propose_team(1), {2,3,4})

    # test that the Naive servant proposes teams of the correct sizes
    def test_propose_team_size(self):
        self.assertEqual(len(self.naive_servant.propose_team(0)), 2)
        self.naive_servant.observe_mission([0, 1], 0, 2)
        self.assertEqual(len(self.naive_servant.propose_team(1)), 3)
        self.naive_servant.observe_mission([0,1,2], 1, 2)
        self.assertEqual(len(self.naive_servant.propose_team(2)), 2)
        self.naive_servant.observe_mission([0,1], 2, 2)
        self.assertEqual(len(self.naive_servant.propose_team(3)), 3)
        self.naive_servant.observe_mission([0,1,2], 3, 2)
        self.assertEqual(len(self.naive_servant.propose_team(4)), 3)

    # test that the Naive servant believes that 2,4 have 3/5 chance of being good after observing team [0,1] fail, players 0,1 have a 2/5 chance of being good, and that 2,4 have 3/5 chance of being good after observing team [0,1,3] fail
    def test_believe_player_sides(self):
        self.naive_servant.observe_mission([0, 1], 0, 1)
        # assert that almost all array elements are equal
        self.assertTrue(np.allclose(self.naive_servant.get_believed_sides(), [0.4, 0.4, 0.6, 1.0, 0.6], atol=0.1))
        self.naive_servant.observe_mission([0, 1, 3], 1, 1)
        self.assertTrue(np.allclose(self.naive_servant.get_believed_sides(), [0.4, 0.4, 0.6, 1.0, 0.6], atol=0.1))
        self.naive_servant.observe_mission([2, 3], 1, 1)
        self.assertTrue(np.allclose(self.naive_servant.get_believed_sides(), [0.5, 0.5, 0, 1.0, 1.0], atol=0.1))
        # print(self.naive_servant.possible_player_sides)
        # print(self.naive_servant.possible_player_sides)
        # print(self.naive_servant.player_side_probabilities)

    # test that the Naive servant proposes subsets and supersets of the last successful team when indifferent
    def test_propose_team_indifferent(self):
        self.naive_servant.observe_mission(frozenset({0,1}), 0, 0)
        self.assertEqual(self.naive_servant.propose_team(1), {0,1,3})
        # self.naive_servant.observe_mission(frozenset({0,1,4}), 1, 1)
        # print(self.naive_servant.propose_team(2))
        # self.assertIn(self.naive_servant.propose_team(1), [{0,1,2}, {0,1,3}, {0,1,4}])
        # self.naive_servant.observe_mission([0, 1, 2], 1, 1)
        # self.assertIn(self.naive_servant.propose_team(2), [{0,1,2}, {0,1,3}, {0,1,4}])
        # self.naive_servant.observe_mission([0, 1, 3], 2, 1)
        # self.assertIn(self.naive_servant.propose_team(3), [{0,1,2}, {0,1,3}, {0,1,4}])
        

if __name__ == '__main__':
    unittest.main()