
class wrap:
    pass


class Bar:

    def __init__(self):

        print('hi from bar')


@wrap
bar = Bar()
