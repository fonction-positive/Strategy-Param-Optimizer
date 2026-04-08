import backtrader as bt
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import argrelextrema
from sklearn.linear_model import QuantileRegressor


# 期货品种配置字典
# 格式说明：
#   name: 品种中文名称
#   exchange: 交易所代码 (SHF: 上期所, DCE: 大商所, CZCE: 郑商所, INE: 能源中心)
#   mult: 合约乘数 (每手对应的标的物数量)
#   margin: 交易所最低保证金比例 (0.1表示10%)
#   comm_type: 手续费类型 (FIXED: 固定金额/手, PERC: 百分比按成交金额)
#   commission: 手续费标准 (FIXED时单位为元/手, PERC时为比例如0.0005表示万分之5)

FUTURES_CONFIG = {
    'SC': {  # 上海国际能源交易中心原油期货
        'name': '原油',
        'exchange': 'INE',
        'mult': 1000,  # 合约乘数: 1000桶/手
        'margin': 0.11,  # 保证金比例: 11%
        'comm_type': 'FIXED',  # 手续费类型: 固定金额
        'commission': 20.0,  # 手续费: 20元/手 (交易所标准)
    },

    'AU': {  # 上海期货交易所黄金期货
        'name': '黄金',
        'exchange': 'SHF',
        'mult': 1000,  # 合约乘数: 1000克/手
        'margin': 0.19,  # 保证金比例: 19%
        'comm_type': 'FIXED',  # 手续费类型: 固定金额
        'commission': 10.0,  # 手续费: 10元/手 (交易所标准)
    },

    'AG': {  # 上海期货交易所白银期货
        'name': '白银',
        'exchange': 'SHF',
        'mult': 15,  # 合约乘数: 15千克/手
        'margin': 0.22,  # 保证金比例: 22%
        'comm_type': 'PERC',  # 手续费类型: 百分比
        'commission': 0.00010,  # 手续费: 万分之1 (0.01%)
    },

    'CU': {  # 上海期货交易所铜期货
        'name': '铜',
        'exchange': 'SHF',
        'mult': 5,  # 合约乘数: 5吨/手
        'margin': 0.12,  # 保证金比例: 10%
        'comm_type': 'PERC',  # 手续费类型: 百分比
        'commission': 0.00050,  # 手续费: 万分之5 (0.05%)
    },

    'LC': {  # 广州期货交易所碳酸锂期货 (Lithium Carbonate)
        'name': '碳酸锂',
        'exchange': 'GFEX',  # 广州期货交易所
        'mult': 1,  # 合约乘数: 1吨/手
        'margin': 0.13,  # 保证金比例: 13%
        'comm_type': 'PERC',  # 手续费类型: 百分比
        'commission': 0.00032,  # 手续费: 万分之3.2 (0.032%)
    },

    'JM': {  # 大商所焦煤期货
        'name': '焦煤',
        'exchange': 'DCE',
        'mult': 60,  # 合约乘数: 60吨/手
        'margin': 0.12,  # 保证金比例: 12%
        'comm_type': 'PERC',
        'commission': 0.00050,  # 手续费: 万分之5 (0.05%)
    },

    'TA': {  # 郑商所 PTA 期货
        'name': 'PTA',
        'exchange': 'CZCE',
        'mult': 5,  # 合约乘数: 5吨/手
        'margin': 0.10,  # 保证金比例: 10%
        'comm_type': 'FIXED',  # 手续费类型: 固定金额
        'commission': 3.0,  # 手续费: 3元/手 (交易所标准)
    },
}


class QuantilePressureSupport(bt.Indicator):
    lines = ('pressure', 'support', 'mid',)
    params = (('lookback', 60), ('window', 5), ('q_pressure', 0.9), ('q_support', 0.1))

    def __init__(self):
        self.addminperiod(self.p.lookback)

    def next(self):
        if len(self.data) < self.p.lookback:
            return

        segment = np.array([self.data.close[i] for i in range(-self.p.lookback, 0)])
        norm_segment = segment
        x = np.arange(len(norm_segment)).reshape(-1, 1)

        high_idx = argrelextrema(norm_segment, np.greater, order=self.p.window)[0]
        low_idx = argrelextrema(norm_segment, np.less, order=self.p.window)[0]

        if len(high_idx) < 2 or len(low_idx) < 2:
            return

        # 修改这里：添加 solver='highs' 参数
        q_reg_high = QuantileRegressor(quantile=self.p.q_pressure, alpha=0, solver='highs')
        q_reg_high.fit(x[high_idx], norm_segment[high_idx])

        q_reg_low = QuantileRegressor(quantile=self.p.q_support, alpha=0, solver='highs')
        q_reg_low.fit(x[low_idx], norm_segment[low_idx])

        current_x = np.array([[len(norm_segment) - 1]])

        self.lines.pressure[0] = q_reg_high.predict(current_x)[0]
        self.lines.support[0] = q_reg_low.predict(current_x)[0]
        self.lines.mid[0] = (self.lines.pressure[0] + self.lines.support[0]) / 2


class KeyLevelStrategy(bt.Strategy):
    params = dict(
        lookback=30,
        window=2,
        q_pressure=0.9,
        q_support=0.3,
        atr_len=28,
        zone_k=0.7,
        reject_n=3,
        wick_ratio=0.6,
        shrink_ratio=0.4,
        cond=2,
        stop_k=1.2,

        contract_multi=1000,
        margin_ratio=0.1,
        cash_percent=0.8,
    )

    def __init__(self):
        super().__init__()

        self.press_support = QuantilePressureSupport(
            lookback=self.p.lookback,
            window=self.p.window,
            q_pressure=self.p.q_pressure,
            q_support=self.p.q_support
        )

        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_len)

        self.order = None
        self.size = 0
        self.last_enter = 0.0

    def confirm_rejection_short(self):
        if len(self.data) < self.p.reject_n:
            return False

        highs = [self.data.high[-i - 1] for i in range(self.p.reject_n)]
        prev_highs = [self.data.high[-i - 2] for i in range(self.p.reject_n)]
        cond1 = all(h <= ph for h, ph in zip(highs, prev_highs))

        if self.data.high[0] != self.data.low[0]:
            wick = (self.data.high[0] - max(self.data.open[0], self.data.close[0])) / (
                    self.data.high[0] - self.data.low[0])
        else:
            wick = 1e6
        cond2 = wick > self.p.wick_ratio

        cond3 = self.data.close[0] < (self.press_support.lines.pressure[0] - self.p.zone_k * self.atr[0])

        vol_avg = np.mean([self.data.volume[-i] for i in range(1, self.p.reject_n + 1)])
        cond4 = self.data.volume[0] < vol_avg * (1 + self.p.shrink_ratio)

        atr_avg = np.mean([self.atr[-i] for i in range(1, self.p.reject_n + 1)])
        cond5 = self.atr[0] < atr_avg * (1 - self.p.shrink_ratio)

        return sum([cond1, cond2, cond3, cond4, cond5]) >= self.p.cond

    def confirm_rejection_long(self):
        if len(self.data) < self.p.reject_n:
            return False

        lows = [self.data.low[-i - 1] for i in range(self.p.reject_n)]
        prev_lows = [self.data.low[-i - 2] for i in range(self.p.reject_n)]
        cond1 = all(l >= pl for l, pl in zip(lows, prev_lows))

        if self.data.high[0] != self.data.low[0]:
            wick = (min(self.data.open[0], self.data.close[0]) - self.data.low[0]) / (
                    self.data.high[0] - self.data.low[0])
        else:
            wick = 1e6
        cond2 = wick > self.p.wick_ratio

        cond3 = self.data.close[0] > (self.press_support.lines.support[0] + self.p.zone_k * self.atr[0])

        vol_avg = np.mean([self.data.volume[-i] for i in range(1, self.p.reject_n + 1)])
        cond4 = self.data.volume[0] < vol_avg * (1 + self.p.shrink_ratio)

        atr_avg = np.mean([self.atr[-i] for i in range(1, self.p.reject_n + 1)])
        cond5 = self.atr[0] < atr_avg * (1 - self.p.shrink_ratio)

        return sum([cond1, cond2, cond3, cond4, cond5]) >= self.p.cond

    def next(self):
        if not self.position:
            close = self.data.close[0]
            atr = self.atr[0]
            press = self.press_support.lines.pressure[0]
            supp = self.press_support.lines.support[0]
            # 可选：调试信息输出
            # print(f"{self.data.datetime.date(0)} - High: {self.data.high[0]:.2f}, Low: {self.data.low[0]:.2f}, "
            #       f"Support: {supp:.2f}, Pressure: {press:.2f}, Zone K: {self.p.zone_k}, ATR: {atr:.2f}")
            
            # 期货仓位计算（修正）
            # self.size = int(self.broker.get_cash() * self.p.cash_percent / close)  # 原股票算法废弃
            # 1. 计算可用资金
            available_cash = self.broker.get_cash() * self.p.cash_percent

            # 2. 计算一手合约的名义价值
            # 名义价值 = 价格 * 合约乘数
            contract_value = close * self.p.contract_multi

            # 3. 计算一手合约需要的保证金
            # 保证金 = 名义价值 * 保证金比例
            margin_per = contract_value * self.p.margin_ratio

            # 4. 计算可以开多少手
            # 手数 = 可用资金 / 每手保证金
            self.size = int(available_cash / margin_per)

            if self.size <= 0:
                return

            if (press - self.p.zone_k * atr <= self.data.high[0] <=
                    press + self.p.zone_k * atr):

                if self.confirm_rejection_short():
                    stop = close + self.p.stop_k * atr
                    tp2 = supp - self.p.zone_k * atr

                    # print(close, stop, tp2)

                    self.last_enter = -close

                    self.order = self.sell_bracket(
                        size=self.size,
                        price=close,
                        stopprice=stop,
                        limitprice=tp2
                    )

            elif (supp - self.p.zone_k * atr <= self.data.low[0] <=
                  supp + self.p.zone_k * atr):

                if self.confirm_rejection_long():
                    stop = close - self.p.stop_k * atr
                    tp2 = press + self.p.zone_k * atr

                    # print(close, stop, tp2)

                    self.last_enter = close

                    self.order = self.buy_bracket(
                        size=self.size,
                        price=close,
                        stopprice=stop,
                        limitprice=tp2
                    )

        # 检查爆仓情况（修正）
        if self.last_enter > 0 and self.position:
            if self.data.close[0] < self.last_enter * (1 - 1 / self.p.cash_percent):
                self.close()

        if self.last_enter < 0 and self.position:
            if self.data.close[0] > (-self.last_enter) * (1 + 1 / self.p.cash_percent):
                self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_enter = 0.0


#  期货手续费类
class FutureCommInfoFixed(bt.CommInfoBase):
    """
    固定金额手续费类（适用于SC原油、AU黄金等）
    """
    params = (
        ('stocklike', False),  # 期货模式（非股票模式）
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # 固定金额模式
        ('commission', 20.0),  # 每手手续费（元）
        ('mult', 1000),  # 合约乘数
        ('margin', 0.15),  # 保证金比例
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算期货手续费
        固定金额模式：手续费 = 交易手数 * 每手手续费
        注意：这里的 size 是手数（已考虑合约乘数）
        """
        return abs(size) * self.p.commission

class FutureCommInfoPerc(bt.CommInfoBase):
    """
    百分比手续费类（适用于AG白银、CU铜等）
    """
    params = (
        ('stocklike', False),
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 百分比手续费模式
        ('percabs', True),  # 绝对百分比
        ('commission', 0.0001),  # 手续费率（如0.0001表示0.01%）
        ('mult', 1),
        ('margin', 0.1),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算期货手续费
        百分比模式：手续费 = 交易手数 * 价格 * 合约乘数 * 手续费率
        注意：这里的 size 是手数，需要乘以合约乘数得到实际交易量
        """
        return abs(size) * price * self.p.mult * self.p.commission
