class BaseManager:

    def __init__(self):
        super().__init__()
        self.model = None
        self.name = None
        self._db = None

#    def
#from .charfield import CharField


# class ModelManager:

#    models = dict()

#    def register_model(ModelCls):

#        assert(ModelCls.__module__.split('.')[-1] == 'models')
#        assert(ModelCls.__module__.split('.')[-1].islower())

#        app_name = ModelCls.__module__.split('.')[-2]

#        assert(app_name.islower())

#        model_name = ModelCls.__name__

#        db_table_name = model_name[0:1].lower()

#        for i in range(1, len(model_name)):
#            if model_name[i:i + 1].isupper():
#                db_table_name += '_' + model_name[i:i + 1].lower()
#            else:
#                db_table_name += model_name[i:i + 1]

#        ModelCls.db_table_name = '_'.join((app_name, db_table_name))

#        Create table columns for model fields.

#        for dkey, dvalue in ModelCls.__dict__.items():
#            if isinstance(dvalue, CharField):
#                print(dvalue.max_length)
#                print(f'{dkey} is a CharField')

#        print(dkey)
