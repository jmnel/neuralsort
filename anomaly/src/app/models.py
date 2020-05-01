import models
from models import Model


class PersonIdentity(Model):

    name = models.CharField()


models.ModelManager.register_model(PersonIdentity)

print(PersonIdentity.db_table_name)
