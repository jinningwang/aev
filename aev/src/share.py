"""Shared Class"""

from collections import OrderedDict
import pandas as pd
import numpy as np

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


class DataUnit():
    """Base class for data unit"""

    def __init__(self, data, col, is_idx=True, idx=None) -> None:
        self.v = data
        self.col = col
        self.is_idx = is_idx
        self.idx = idx

    def as_df(self) -> pd.DataFrame:
        """Return data as a pandas.DataFrame"""
        if self.is_idx:
            pass
            df = pd.DataFrame()
            df['idx'] = self.idx
            df[self.col] = self.v
        else:
            df = pd.DataFrame(data=self.v, columns=self.col)
        return df

    def set(self, src, data, cond=None) -> bool:
        """
        Set value to a column.

        Parameters
        ----------
        src: str
            Column name.
        data: np.array
            Data to be set.
        cond: np.array, optional
            An array of bool, condition to set data.
            ``data`` must have the length of the length of ``True``.

            Only positions where ``cond`` is ``True`` will be set.

        Example
        -------
        ``cond = (self.v[:, col_idx] == 0)``

        ``set(src='u', data=u_data, cond=cond)``
        """
        col_idx = self.col.index(src)
        if cond is not None:
            if np.sum(cond) != len(data):
                raise ValueError(
                    f'Data shape incompatible: data length: {len(data)}, True cond: {np.sum(cond)}')
            self.v[:, col_idx][cond] = data
        else:
            self.v[:, col_idx] = data
        return True

    def get(self, src, cond=None) -> np.array:
        """
        Get the copy of value of a column

        Parameters
        ----------
        src: str
            Column name.

        cond: np.array, optional
            An array of bool, condition to access data.
            The length of ``cond`` must be the same as the length of the accessed data.

            Only positions where ``cond`` is ``True`` will be retrieved.
        """
        col_idx = self.col.index(src)
        if cond is not None:
            if len(cond) != self.v.shape[0]:
                raise ValueError(
                    f'Data shape incompatible: data length: {self.v.shape[0]}, True cond: {np.sum(cond)}')
        return self.v[:, col_idx].copy()
