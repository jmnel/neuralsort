class IBMockAccount:

    def __init__(self,
                 init_funds=10000,
                 fee_min=1.00,
                 fee_max_percentage=0.005,
                 fee_per_share=0.005,
                 init_portfolio=None):

        self.init_funds = init_funds
        self.fee_min = fee_min
        self.fee_max_percentage = fee_max_percentage
        self.fee_per_share = fee_per_share

        if init_portfolio is not None:
            self.portfolio = init_portfolio
        else:
            self.portfolio = dict()

    def place_bid_market_order(symbol, price, size):

        assert symbol in self.portfolio
        assert self.portfolio[symbol] >= size

    def place_ask_market_order(symbol, price, size):
        pass
