from .model import Model
from .charfield import CharField
from .autofield import AutoField
from .manager import ModelManager


class Field:

    def __init__(self, **options):

        # null option
        if 'null' in options:
            self.null = options['null']
            if not isinstance(self.null, bool):
                raise ValueError(
                    'Field option `null` must be either True or False')
        else:
            self.null = False

        # blank option
        if 'blank' in options:
            self.blank = options['blank']
            if not isinstance(self.blank, bool):
                raise ValueError(
                    'Field option `blank` must be either True or False')
        else:
            self.blank = False

        # choices option
        if 'choices' in options:
            self.choices = options['choices']
            if not isinstance(self.choices, tuple) and not isinstance(self.choices, list):
                raise ValueError(
                    'Field option `choices` must be list or tupple')

            for c in self.choices:
                if not isinstance(c, tuple) and not isinstance(c, list):
                    raise ValueError(
                        'Field option `choices` elements must be tuple or list')

                if len(c) != 2:
                    raise ValueError(
                        'Field option `choices` elements must be length 2')

        else:
            self.choices = None

        # unique option
        if 'unique' in options:
            self.unique = options['unique']
            if not isinstance(self.unique, bool):
                raise ValueError(
                    'Field option `unique` must be either True or False')
        else:
            self.unique = False
