def notify(subscribers, observation):
    for sub in subscribers:
        sub.on_observation(observation)


class ObservationSubscriber(object):
    def on_observation(self, observation):
        print("Not implemented observation")
