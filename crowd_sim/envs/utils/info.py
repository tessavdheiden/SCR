class Info(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Info'


class Timeout(Info):
    def __init__(self):
        super(Info, self).__init__()
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(Info):
    def __init__(self):
        super(Info, self).__init__()
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(Info):
    def __init__(self, min_dist):
        super(Info, self).__init__()
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision(Info):
    def __init__(self):
        super(Info, self).__init__()
        pass

    def __str__(self):
        return 'Collision'


class Nothing(Info):
    def __init__(self):
        super(Info, self).__init__()
        pass

    def __str__(self):
        return ''
