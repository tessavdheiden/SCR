from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class Interactive(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        # hard-coded keyboard events
        self.move = [False for i in range(4)]

    def configure(self, config):
        assert True

    def predict(self, state):
        vx = 0
        vy = 0
        if self.move[0]: vx = -1
        if self.move[1]: vx = 1
        if self.move[2]: vy = 1
        if self.move[3]: vy = -1
        action = ActionXY(vx, vy)
        return action

    # keyboard event callbacks
    def key_press(self, k):
        if k=='left':  self.move[0] = True; self.move[1] = False
        if k=='right': self.move[1] = True; self.move[0] = False
        if k=='up':    self.move[2] = True; self.move[3] = False
        if k=='down':  self.move[3] = True; self.move[2] = False
    def key_release(self, k):
        if k=='left':  self.move[0] = False
        if k=='right': self.move[1] = False
        if k=='up':    self.move[2] = False
        if k=='down':  self.move[3] = False
