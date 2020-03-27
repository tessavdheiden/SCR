

class Action(object):
    pass

class ActionXY(Action):
    def __init__(self, vx, vy):
        self.vx = vx
        self.vy = vy

class ActionRot(Action):
    def __init__(self, v, r):
        self.v = v
        self.r = r

