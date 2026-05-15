import pygame
from argparse import ArgumentParser

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner


KEYS = {
    'UP': pygame.K_UP,
    'RIGHT': pygame.K_RIGHT,
    'DOWN': pygame.K_DOWN,
    'LEFT': pygame.K_LEFT,
    'SPACE': pygame.K_SPACE,
}


class App:
    """Run an Overcooked Gridworld game with one human player and one scripted agent."""
    def __init__(self, env, agent, human_idx=0, slow_time=True):
        self.env = env
        self.agent = agent
        self.human_idx = int(human_idx)
        self.slow_time = slow_time
        self._running = True

    def on_init(self):
        pygame.init()
        # attach agent to mdp
        try:
            self.agent.set_agent_index(1 - self.human_idx)
            self.agent.set_mdp(self.env.mdp)
        except Exception:
            pass
        self._running = True

    def on_event(self, event):
        done = False
        if event.type == pygame.KEYDOWN:
            key = event.key
            action = None
            if key == KEYS['UP']:
                action = Direction.NORTH
            elif key == KEYS['RIGHT']:
                action = Direction.EAST
            elif key == KEYS['DOWN']:
                action = Direction.SOUTH
            elif key == KEYS['LEFT']:
                action = Direction.WEST
            elif key == KEYS['SPACE']:
                action = Action.INTERACT

            if action is not None:
                done = self.step_env(action)
                if self.slow_time and not done:
                    # insert a stay step to slow down
                    done = self.step_env(Action.STAY)

        if event.type == pygame.QUIT or done:
            print("Total sparse reward:", self.env.cumulative_sparse_rewards)
            self._running = False

    def step_env(self, human_action):
        # get scripted agent action
        try:
            scripted_action = self.agent.action(self.env.state)
        except Exception:
            scripted_action = Action.STAY

        if self.human_idx == 0:
            joint = (human_action, scripted_action)
        else:
            joint = (scripted_action, human_action)

        s, r, done, info = self.env.step(joint)
        print(self.env)
        print("Reward (sparse):", r, " shaped:", info.get('shaped_r'))
        return done

    def on_loop(self):
        pass

    def on_render(self):
        pass

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if not self.on_init():
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
        self.on_cleanup()


def setup_game(agent_type, layout_name, player_idx, horizon=400):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv(mdp, horizon=horizon)

    agent_type = agent_type.lower()
    if agent_type == 'random':
        agent = RandomAgent()
    elif agent_type == 'stay':
        agent = StayAgent()
    elif agent_type == 'planner':
        # medium level planner expects mdp set later; create with default params
        agent = MediumLevelPlanner(mdp)
    elif agent_type == 'greedy':
        agent = GreedyHumanModel()
    else:
        raise ValueError('Unsupported agent type: {}'.format(agent_type))

    return env, agent, player_idx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', dest='type', required=False, default='random',
                        help='agent type for teammate: random, stay, planner, greedy')
    parser.add_argument('-l', '--layout', dest='layout', default='simple', help='layout name')
    parser.add_argument('-i', '--idx', dest='idx', default=1, help='human player index (0 or 1)')
    parser.add_argument('-s', '--slow', dest='slow', action='store_true', help='slow mode')
    args = parser.parse_args()

    env, agent, player_idx = setup_game(args.type, args.layout, args.idx)
    app = App(env, agent, player_idx, slow_time=args.slow)
    print('Human player index:', player_idx)
    app.on_execute()
