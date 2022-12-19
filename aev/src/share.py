"""Shared Class"""

from collections import OrderedDict
import pprint


def safe_div(x, y):
    '''
    Safe division, return 0 if y is 0.

    Parameters
    ----------
    x: float
        numerator.
    y: float
        denominator.
    '''
    if y == 0:
        return 0
    else:
        return x/y


class DictAttr():
    """Class for attributes stored in OrderedDict"""

    def __init__(self, attr):
        """
        Base class for attribtues stored in OrderedDict

        Parameters
        ----------
        attr: OrderedDict
            Data attribute dictionary
        """
        for key, val in attr.items():
            setattr(self, key, val)
        self._dict = self.as_dict()

    def as_dict(self) -> OrderedDict:
        """
        Return the config fields and values in an ``OrderedDict``.
        """
        out = []
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                out.append((key, val))
        return OrderedDict(out)

    def __repr__(self):
        self._dict = self.as_dict()
        return pprint.pformat(self._dict)
