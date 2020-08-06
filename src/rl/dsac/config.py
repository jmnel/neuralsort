class Config():
    """
    Data structure to hold parameters for agent and environemtn.

    """

    def __init__(self):
        self.seed = None
        self.environment = None
        self.num_episodes_to_run = None
        self.runs_per_aganet = None
        self.hyperameters = None
        self.use_gpu = None
