class ModelBase(type):

    def __new__(cls, name, bases, attrs, **kwargs):
        super_new = super().__new__


#    def __init__(self):
#        print('model base created')
