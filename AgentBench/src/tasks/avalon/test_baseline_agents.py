from baseline_agents import *
import unittest
from engine import AvalonConfig, AvalonGameEnvironment

# Test the baseline agents. Use python unittest module.
class TestNaiveServant(unittest.TestCase):
    
    # create Naive Servant
    def setUp(self):
        config = AvalonConfig(5)
        self.env = AvalonGameEnvironment(config)
        self.naive_servant = NaiveServant(3, 'Naive Servant', config, sides=self.env.get_partial_sides)

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
        self.naive_servant.observe_mission([0, 1], 0, 2)
    
    # test that the Naive servant rejects any team that is superset of the team that failed when using vote_on_team
    def test_vote_on_team_superset(self):
        self.naive_servant.observe_mission([0, 1], 0, 2)
        self.assertEqual(self.naive_servant.vote_on_team(1, [0, 1, 3]), 0)

    # test that the Naive servant only proposes a team of only good players after observing a team fail, ie. the team [2,3,4]
    def test_propose_team_after_fail(self):
        self.naive_servant.observe_mission([0, 1], 0, 2)
        self.assertEqual(self.naive_servant.propose_team(1), (2, 3, 4))

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

if __name__ == '__main__':
    unittest.main()