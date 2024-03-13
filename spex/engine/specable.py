from copy import deepcopy

from specable import Specable as _Specable


class Specable(_Specable):
    def get_dict(self):
        return deepcopy(self.spec)

    @classmethod
    def get_kind(cls):
        return cls.__name__
