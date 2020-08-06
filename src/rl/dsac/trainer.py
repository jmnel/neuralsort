class Trainer():

    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        self.results = None

    def run_agent(self):
