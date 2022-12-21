from aev.src.share import DictAttr
from tqdm import tqdm  # NOQA
import pandas as pd
import numpy as np
import os
import scipy.stats as stats  # NOQA
import matplotlib.pyplot as plt  # NOQA

import logging
logger = logging.getLogger(__name__)


class EVData():
    """EV data class"""

    def __init__(self, idx, cols, N) -> None:
        """
        Two pandas.DataFrame are used to store EV data, one for static data,
        the other for dynamic data.

        Static data is initialized once, and dynamic data is updated at each
        time step by ``MCS.run()``.

        After initialization, Static data is not allowed to be changed.

        Parameters
        ----------
        idx: list
            EV index.
        cols: list
            EV data columns.
        N: int
            Number of rows.
        """
        self.idx = idx
        self._cols = cols
        for col in self._cols:
            setattr(self, col, np.array([-9.0]*N))

    def as_df(self) -> pd.DataFrame:
        """Convert to pandas.DataFrame"""
        # TODO: improve
        df = pd.DataFrame()
        df['idx'] = self.idx
        for col in self._cols:
            df[col] = getattr(self, col)
        return df


class EVS():
    """
    EV Station class to hold EV data, control EV status, and collecte EV info.
    """

    def __init__(self,
                 config, mcs_config,
                 ud_param, nd_param,
                 name='EVS') -> None:
        """
        Parameters
        ----------
        config: dict
            EV station configuration.
        ud_param: Dict of Dict
            Uniform distribution parameters.
        nd_param: Dict of Dict
            Normal distribution parameters.
        t: float
            Simulation start time in 24 hour.
        name: str
            EV station name.

        config
        ------
        N: int
            Number of EVs
        Ns: int
            Number of SOC intervals
        Tagc: float
            AGC time period in second
        socf: float
            SOC level that will be switched to force charging
        seed: int
            random seed
        r: float
            Ratio of time param type1 to type2, [0, 1].

        nd_param
        --------
        soci: float
            Initial SOC
        socd: float
            Demanded SOC
        ts1: float
            Start charging time 1
        ts2: float
            Start charging time 2
        tf1: float
            Finish charging time 1
        tf2: float
            Finish charging time 2
        tt: float
            Tolerance of increased charging time

        ud_param
        --------
        Pc: float
            Rated charging power
        Pd: float
            Rated discharging power
        nc: float
            Rated charging efficiency
        nd: floar
            Rated discharging efficiency
        Q: float
            Rated battery capacity
        """
        self.name = name
        # --- config ---
        self.config = DictAttr(config)
        # --- declear data variable ---
        idx = [f'{name}_{i}' for i in range(self.config.N)]
        cols = [
            # --- dynamic ---
            'u', 'u0', 'soc', 'c', 'lc', 'sx',
            'na', 'ama', 'agc', 'mod',
            # --- static ---
            'nam', 'ts', 'tf', 'tt', 'soc0', 'na0',
            'soci', 'socd', 'Pc', 'Pd',
            'nc', 'nd', 'Q']

        # --- initialize MCS ---
        # add current timestamp `t` to config
        mcs_config = {**{'t': 0.0, 'tf': 0.0,
                         'socf': self.config.socf,
                         'agc': self.config.agc,
                         'ict': self.config.ict},
                      **mcs_config}
        # put data into MCS
        self.MCS = MCS(config=mcs_config, idx=idx,
                       cols=cols, N=self.config.N)
        mdp = self.MCS.data  # pointer to MCS data
        # TODO: CODE SMELL: declear in ``MCS`` but initialize here

        # --- initialize data ---
        # --- 1. uniform distribution parameters ---
        ud_cols = ['Pc', 'Pd', 'nc', 'nd', 'Q']
        np.random.seed(self.config.seed)
        for col in ud_cols:
            value = np.random.uniform(size=self.config.N,
                                      low=ud_param[col]['lb'],
                                      high=ud_param[col]['ub'])
            setattr(mdp, col, value)
        mdp.nd = mdp.nc  # NOTE: assumtpion: nc = nd

        # --- 2. normal distribution parameters ---
        # --- 2.1 non-time parameters ---
        nd_cols = ['soci', 'socd', 'tt']
        for col in nd_cols:
            a = (nd_param[col]['lb'] - nd_param[col]
                 ['mu']) / nd_param[col]['var']
            b = (nd_param[col]['ub'] - nd_param[col]
                 ['mu']) / nd_param[col]['var']
            distribution = stats.truncnorm(a, b,
                                           loc=nd_param[col]['mu'],
                                           scale=nd_param[col]['var'])
            value = distribution.rvs(self.config.N,
                                     random_state=self.config.seed)
            setattr(mdp, col, value)

        # --- 2.2 time parameters ---
        nd_cols = ['ts1', 'ts2', 'tf1', 'tf2']
        tparam = pd.DataFrame()
        for col in nd_cols:
            a = (nd_param[col]['lb'] - nd_param[col]
                 ['mu']) / nd_param[col]['var']
            b = (nd_param[col]['ub'] - nd_param[col]
                 ['mu']) / nd_param[col]['var']
            distribution = stats.truncnorm(a, b,
                                           loc=nd_param[col]['mu'],
                                           scale=nd_param[col]['var'])
            tparam[col] = distribution.rvs(self.config.N,
                                           random_state=self.config.seed)

        r = self.config.r  # ratio of ts1 to ts2
        tp1 = tparam[['ts1', 'tf1']].sample(n=int(self.config.N * r),
                                            random_state=self.config.seed)
        tp2 = tparam[['ts2', 'tf2']].sample(n=int(self.config.N * (1 - r)),
                                            random_state=self.config.seed)
        tp = pd.concat([tp1, tp2], axis=0).reset_index(drop=True).fillna(0)
        tp['ts'] = tp['ts1'] + tp['ts2']
        tp['tf'] = tp['tf1'] + tp['tf2']

        check = tp['ts'] > tp.tf
        row_idx = tp[check].index
        mid = tp['tf'].iloc[row_idx].values
        tp['tf'].iloc[row_idx] = tp['ts'].iloc[row_idx]
        tp['ts'].iloc[row_idx] = mid
        setattr(mdp, 'tf', tp['tf'].values)
        setattr(mdp, 'ts', tp['ts'].values)

        # --- memory save ---
        if self.config.memory_save:
            mask_u1 = mdp.ts > (self.MCS.config.ts + self.MCS.config.th)
            mask_u2 = mdp.tf < self.MCS.config.ts
            mask = mask_u1 | mask_u2
            drop_id = np.where(mask)[0]
            for col in mdp._cols + ['idx']:
                value = getattr(mdp, col)
                value = np.delete(value, drop_id, axis=0)
                setattr(mdp, col, value)
            # --- info ---
            info_mem_save = f'Memory save:, EVs out of time range '\
                f'[{self.MCS.config.ts}, {self.MCS.config.ts + self.MCS.config.th}] are dropped.'
            logger.warning(info_mem_save)

        # # --- 3. online status ---
        self.MCS.g_u()
        mdp.u0 = mdp.u.copy()

        # --- 4. initialize SOC ---
        # TODO: do we need to consider the AGC participation?
        # time required to charge to demanded SOC
        tr = (mdp.socd - mdp.soci) * mdp.Q / mdp.Pc / mdp.nc
        # time that has charged
        tc = self.MCS.config.ts - mdp.ts
        tc[tc < 0] = 0  # reset negative time to 0
        # charge
        mdp.soc = mdp.soci + tc * mdp.Pc * mdp.nc / mdp.Q
        kt = tc / tr  # ratio of stay/required time
        kt[kt < 1] = 1
        # higher than required charging time, log scale higher than socd
        socp = mdp.socd + np.log(kt) * (1 - mdp.socd)
        socp[socp > 1.0] = 1.0
        socp[socp < 0.0] = 0.0
        mdp.soc[kt > 1] = socp[kt > 1]  # reset ``soc`` that are out of range
        mdp.soc0 = mdp.soc.copy()  # initialize ``soc0``
        mdp.sx = np.ceil(mdp.soc / (1 / self.config.Ns)) - \
            1  # initialize soc level ``sx

        # --- 5. Control, AGC, and mod ---
        mask = (mdp.soc < mdp.socd) & (mdp.u > 0)
        mdp.c = np.float64(mask)  # EV Control ``c``
        mdp.agc = np.zeros(len(mdp.idx))  # EV AGC indicator ``agc``
        mdp.mod = np.zeros(len(mdp.idx))  # EV MOD indicator ``mod``

        # # --- 6. initialize na [number of action] ---
        # TODO: this part might need improvement
        # history data of ina
        if self.config.ict:
            ina = np.genfromtxt(
                os.getcwd() + '/aev/data/ev_ina_ict.csv', delimiter=',')
        else:
            ina = np.genfromtxt(
                os.getcwd() + '/aev/data/ev_ina.csv', delimiter=',')

        # initialization of number of actions; truncated normal distribution;
        sx0 = np.ceil(mdp.soc / (1 / self.config.Ns)) - 1
        # size of each soc level
        sx0 = pd.Series(sx0)
        sx0d = sx0.value_counts().sort_index()
        for i in sx0d.index:
            i = int(i)
            a, b = ina[i, 2], ina[i, 3]
            if a == b:
                b = a + 0.01
            pdf = stats.norm(loc=0, scale=ina[i, 1])
            res = pdf.rvs(sx0d[float(i)],
                          random_state=self.config.seed).round(0)
            mask = (mdp.sx == i)  # EVs that are in the given soc level
            mdp.na[mask] = ina[i, 0] * \
                (mdp.tf[mask] - self.MCS.config.ts) + res
        mask = (mdp.soc < mdp.socd) & (mdp.na < 0)
        na0_data = 1000 * (mdp.socd - mdp.soc)
        mdp.na[mask] = na0_data[mask]
        # TODO: DEBUG: scale up soc [0.6, 0.7] na0
        mask = (mdp.soc < 0.7) & (mdp.soc > 0.5)
        mdp.na[mask] *= 10
        # for fully charged EVs, reset their na to 0
        mdp.na[mdp.soc >= mdp.socd] = 0.0
        mdp.na0 = mdp.na.copy()
        # TODO: calc number of action mileage
        # `nama` is the number of action mileage
        mdp.ama = np.zeros(len(mdp.idx))
        mdp.na[mdp.na < 0.0] = 0.0

        # --- 8. initialize nam [max number of action] ---
        pcn = mdp.Pc * mdp.nc
        nam_num = (mdp.tf - mdp.ts + mdp.tt) * pcn - mdp.socd * mdp.Q
        nam_den = pcn * self.config.Tagc / 3600
        mdp.nam = nam_num / nam_den
        mdp.nam = mdp.nam.round(0).astype(float)

        # --- 9. initialize lc ---
        mdp.lc = np.zeros(len(mdp.idx))
        if self.config.ict:
            mask = (mdp.na > mdp.nam)  # EVs that `na` exceeds `nam`
            mdp.na[mask] = mdp.nam[mask]
            mdp.lc[mask] = 1.0
        mask = mdp.soc <= self.config.socf  # force charging SOC level
        mdp.lc[mask] = 1.0

        # --- 10. initialize MCS data ---
        self.MCS.g_ts()

        # --- 11. data dict ---
        # TODO: how to organize?
        # NOTE: include MCS info and online EV info

        # --- report initialization info ---
        init_info = f'{self.name}: Initialized successfully with:\n'\
            f'Capacity: {self.config.N}, r: {self.config.r}\n'\
            + self.__repr__()
        logger.warning(init_info)

    def __repr__(self) -> str:
        # TODO: how to organize?
        total = len(self.MCS.data.idx)
        online = int(self.MCS.data.u.sum())
        info = f'{self.name}: Clock: {self.MCS.config.ts + self.MCS.config.t / 3600}[H], Online: {online}, Total: {total}'
        return info

    def rctrl(self):
        """Response to control signal"""
        # TODO: response to control signal
        pass


class MCS():
    """
    Class for Monte-Carlo simulation.
    Store EV data and timeseries data.
    """

    def __init__(self, config, idx, cols, N) -> None:
        """
        Parameters
        ----------
        config: dict
            Monte-Carlo simulation configuration.
        idx: list
            Index of the EV.
        cols: list
            Columns of the EV data.
        N: int
            Number of EVs.

        config
        ------
        t: float
            Current timestamp in seconds.
        tf: float
            Simulation end time in seconds.
        ts: float
            Simulation start time in 24 hour.
        th: float
            Simulation tiem horizon time in hour.
            If memory_save is True, all the EVs out of the
            time horizon ``[ts, ts + th]`` will be dropped.
        h: float
            Simulation time step in second.
        no_tqdm: bool
            Disable tqdm progress bar.
        """
        self.config = DictAttr(config)

        # --- EV data ---
        self.data = EVData(idx=idx, cols=cols, N=N)
        # TODO: CODE SMELL: declear here but initialize in ``EVS``

        # --- Time series data ---
        # --- 1. Station data ---
        # NOTE: this might need to be extended if sim time is longer than ``th``
        # TODO: ask Hantao about dynamic variable declaration
        ts_cols = ['t', 'Pi', 'Prc', 'Ptc']
        t = np.arange(0, self.config.th * 3600 + 0.1,
                      self.config.h)  # timestamp in seconds
        self.ts = EVData(idx=range(len(t)),  # CODESMELL: ``range(len(t))`` is not a list
                         N=len(t), cols=ts_cols)
        for col in ts_cols:
            setattr(self.ts, col, np.zeros(len(t)))
        self.ts.t = t
        # --- 2. single EV data with part dynamic columns ---
        # TODO: might include the dynamic data in the future
        # dts_cols = ['u', 'soc', 'c', 'lc', 'sx', 'na', 'agc', 'mod']
        # self.tsu = EVData(idx=[range(N)], N=N, cols=idx)

        # --- Declear info dict ---
        info = {'t': self.config.t, 'Pi': 0,
                'Prc': 0.0, 'Ptc': 0}
        self.info = DictAttr(info)

    def g_u(self) -> bool:
        """
        Update EV online status.
        """
        # --- variable pointer ---
        mdp = self.data  # pointer to ``MCS``(``self``) data
        mdp.u0 = mdp.u  # log previous online status
        # --- check time range ---
        u_check1 = mdp.ts <= self.config.ts
        u_check2 = mdp.tf >= self.config.ts
        u_check = u_check1 & u_check2
        # --- update value ---
        mdp.u = np.array(u_check, dtype=float)
        return mdp.u

    def g_ts(self) -> True:
        """Update info into time series data"""
        mdp = self.data  # pointer to ``data`` of ``MCS``
        # --- calculate charging station power ---
        # NOTE: ``Ptc``, ``Prc``, from kW to MW, seen from the grid
        Prc = mdp.agc * mdp.u * mdp.Pc * mdp.nc
        Ptc = mdp.c * mdp.u * mdp.Pc * mdp.nc
        info = {'t': self.config.t, 'Pi': 0,
                'Prc': -1 * Prc.sum() * 1e-3,
                'Ptc': -1 * Ptc.sum() * 1e-3}
        self.info = DictAttr(info)
        mtsp = self.ts  # pointer to time series data ``ts`` of ``MCS``
        # --- update time series data ---
        diff_t = mtsp.t - self.config.t  # time difference
        # TODO: this is hard coded
        # set negative time difference to a large number
        diff_t[diff_t < 0] = 99999.0
        # find the index of minimum element from the array
        index = diff_t.argmin()
        mtsp.Pi[index] = info['Pi']
        mtsp.Prc[index] = info['Prc']
        mtsp.Ptc[index] = info['Ptc']
        return True

    def __repr__(self) -> str:
        # TODO; any other info?
        t0 = self.ts.t[0]
        t1 = self.config.tf
        info = f'MCS: start from {t0}s, end at {t1}s. Clock begins from {self.config.ts}[H]'
        return info

    # def run(self) -> bool:
    #     """
    #     Run Monte-Carlo simulation
    #     """
    #     # TODO: extend variable if self.config.tf > self.config.th * self.config.h
    #     # --- variable ---
    #     datas = self.data.s
    #     datad = self.data.d
    #     t0 = self.config.t  # start time of this run
    #     resume = t0 > 0  # resume flag, true means not start from zero
    #     perc_t = 0  # total percentage
    #     perc_add = 0  # incremental percentage
    #     pbar = tqdm(total=100, unit='%', file=sys.stdout,
    #                 disable=self.config.no_tqdm)
    #     # --- loop ---
    #     while self.config.t < self.config.tf:
    #         # --- computation ---
    #         # --- 1. update timestamp ---
    #         self.config.t += self.config.h

    #         # --- 2. update EV online status ---
    #         self.g_u()

    #         # --- 3. update control ---
    #         self.g_c()
    #         # --- 4. update EV dynamic data ---
    #         # --- 4.1 update soc interval and online status ---
    #         # charging/discharging power, kW
    #         datad['soc'] += datad['c'] * datas['nc'] * datas['Pc'] \
    #             / datas['Q'] * self.config.h / 3600
    #         # --- 4.2 modify outranged SoC ---
    #         masku = datad[datad['soc'] >= 1.0].index
    #         maskl = datad[datad['soc'] <= 0.0].index
    #         datad.loc[masku, 'soc'] = 1.0
    #         datad.loc[maskl, 'soc'] = 0.0

    #         # --- log info ---
    #         self.g_ts()
    #        # --- 2. update progress bar ---
    #         if resume:
    #             perc = 100 * self.config.h / (self.config.tf - t0)
    #             perc_add = 100 * t0 / self.config.tf
    #             resume = False  # reset resume flag
    #         else:
    #             perc = 100 * self.config.h / self.config.tf
    #         perc_update = perc + perc_add
    #         perc_update = round(perc_update, 2)
    #         # --- limit pbar not exceed 100 ---
    #         perc_update = min(100 - perc_t, perc_update)
    #         pbar.update(perc_update)
    #         perc_t += perc_update

    #     pbar.close()
    #     # TODO: exit_code
    #     return True

    def g_c(self, cvec=None, is_test=False) -> bool:
        """
        Generate EV control signal.
        EV start to charge with rated power as soon as it plugs in.
        The process will not be interrupted until receive control signal
        or achieved demanmded SoC level.

        Test mode is used to build SSM A.

        Parameters
        ----------
        cvec: np.array
            EV control vector from EVCenter
        is_test: bool
            `True` to turn on test mode.
            test mode: TBD
        """
        mdp = self.data  # pointer to ``MCS``(``self``) data
        if is_test:
            # --- test mode ---
            # TODO: add test mode
            return True
        if cvec:
            # --- response with control vector ---
            # TODO: if control not zero, response to control signal
            # NOTE: from EVC
            pass
            return True
        else:
            # --- revise control if no signal ---
            # `CS` for low charged EVs, and set 'lc' to 1
            mask_lc = (mdp.soc <= self.config.socf) & (mdp.u >= 1.0)
            mdp.c[mask_lc] = 1.0
            mdp.mod[mask_lc] = 1.0
            # `IS` for full EVs
            mask_full = mdp.soc >= mdp.socd
            mdp.c[mask_full] = 0.0
            # `IS` for offline EVs
            mdp.c *= mdp.u
            return mdp.c
