import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
class BackTest(object):
    def __init__(self, close_series: pd.Series, singal_ser: pd.Series):

        self.close_ser = close_series

        self.singal = singal_ser
        self.algorithm_ret = pd.Series()
        self.nav_df = self.GetAlgorithm()

    def GetAlgorithm(self):

        next_ret = self.close_ser.pct_change().shift(-1).reindex(
            self.singal.index[:-1])
        
        
        self.algorithm_ret = next_ret * self.singal.iloc[:-1]
        algorithm_cum = (1 + self.algorithm_ret).cumprod()
        
        benchmark = (self.close_ser / self.close_ser[0]).reindex(
            self.singal.index[:-1])

        return pd.DataFrame({
            'benchmark': benchmark,
            'algorithm_cum': algorithm_cum
        })
    

    def plot_algorithm_cum(self):
        
        
        plt.figure(figsize=(18, 8))
        plt.title('策略净值')
        plt.plot(
            self.nav_df['benchmark'],
            color='black',
            alpha=0.7,
            label='BENCHMARK')
        plt.plot(
            self.nav_df['algorithm_cum'],
            color='red',
            alpha=0.7,
            label='ALGORITHM_CUM',
            markevery=self._GetMaxDrawdown(self.nav_df),
            marker='^',
            mec='black',
            mew=2,
            mfc='g',
            ms=7)
        plt.legend()
        plt.show()

    @property
    def GetRisk(self, show=True):
        '''
        当show=Fals输出的表格
        1.风险收益汇总表;2.分年度收益汇总表
        '''
        if show:
            print('风险指标')
            print(self.strategy_performance())
            print('分年度风险指标')
            print(self.get_return_year('algorithm_cum'))

        else:

            return self.strategy_performance(), self.get_return_year(
                'algorithm_cum')

    #计算组合收益率分析:年化收益率、收益波动率、夏普比率、最大回撤
    def strategy_performance(self, nav_df=None):

        if isinstance(nav_df, pd.DataFrame):

            nav_df = nav_df
        else:
            nav_df = self.nav_df

        ##part1:根据回测净值计算相关指标的数据准备（日度数据）
        nav_next = nav_df.shift(1)
        return_df = (nav_df - nav_next) / nav_next  #计算净值变化率，即为日收益率,包含组合与基准
        return_df = return_df.dropna()  #在计算净值变化率时，首日得到的是缺失值，需将其删除

        analyze = pd.DataFrame()  #用于存储计算的指标

        ##part2:计算年化收益率
        cum_return = np.exp(np.log1p(return_df).cumsum()) - 1  #计算整个回测期内的复利收益率
        annual_return_df = (1 + cum_return)**(252 /
                                              len(return_df)) - 1  #计算年化收益率
        analyze['annual_return'] = annual_return_df.iloc[
            -1]  #将年化收益率的Series赋值给数据框

        #part3:计算收益波动率（以年为基准）
        analyze['return_volatility'] = return_df.std() * np.sqrt(
            252)  #return中的收益率为日收益率，所以计算波动率转化为年时，需要乘上np.sqrt(252)

        #part4:计算夏普比率
        risk_free = 0
        return_risk_adj = return_df - risk_free
        analyze['sharpe_ratio'] = return_risk_adj.mean() / np.std(
            return_risk_adj, ddof=1)

        #prat5:计算最大回撤
        cumulative = np.exp(np.log1p(return_df).cumsum()) * 100  #计算累计收益率
        max_return = cumulative.cummax()  #计算累计收益率的在各个时间段的最大值
        analyze['max_drawdown'] = cumulative.sub(max_return).div(
            max_return).min()  #最大回撤一般小于0，越小，说明离1越远，各时间点与最大收益的差距越大

        #part6:计算相对指标
        analyze['relative_return'] = analyze['annual_return'] - analyze.loc[
            'benchmark', 'annual_return']  #计算相对年化波动率
        analyze['relative_volatility'] = analyze[
            'return_volatility'] - analyze.loc['benchmark',
                                               'return_volatility']  #计算相对波动
        analyze['relative_drawdown'] = analyze['max_drawdown'] - analyze.loc[
            'benchmark', 'max_drawdown']  #计算相对最大回撤

        #part6:计算信息比率
        return_diff = return_df.sub(
            return_df['benchmark'], axis=0).std() * np.sqrt(
                252)  #计算策略与基准日收益差值的年化标准差
        
        analyze['info_ratio'] = analyze['relative_return'].div(return_diff)

        analyze = analyze.T
        analyze = analyze.append(self.TradeCount,sort=True)

        return analyze

    # 构建每年的收益表现函数
    # get_return_year(nav[['benchmark','xxxx']])
    def get_return_year(self, method):

        nav = self.nav_df[['benchmark', method]]

        result_dic = {}  #用于存储每年计算的各项指标
        for y, nav_df in nav.groupby(pd.Grouper(level=0, freq='Y')):

            result = self.strategy_performance(nav_df)
            result_dic[str(y)[:4]] = result.iloc[:-3, 0]

        result_df = pd.DataFrame(result_dic)

        return result_df.T
    
    @staticmethod
    # 获取最大回撤
    def _GetMaxDrawdown(nav_df:pd.DataFrame) -> tuple:
        '''
        algorithm_cum
        ===========
        return 最大回撤位置,最大回撤
        '''
        print(nav_df)
        arr = nav_df['algorithm_cum'].values
        i = np.argmax((np.maximum.accumulate(arr) - arr) /
                      np.maximum.accumulate(arr))  # end of the period
        j = np.argmax(arr[:i])  # start of period
        
        return [i, j]  # 回撤点，回撤比率

    @property
    def TradeCount(self):

        return pd.DataFrame(
            self._GetWinCount(),
            index=['trade_count', 'win_count', 'win_ratio'],
            columns=['algorithm_cum'])

    # TradeCound的底层计算
    def _GetWinCount(self) -> list:
        '''
        统计
        '''

        flag = np.abs(self.singal.values[:-1])
        # 交易次数 1为开仓 -1为平仓
        trade_num = flag - np.insert(flag[:-1], 0, 0)

        # 交易次数
        open_num = sum(trade_num[trade_num == 1])

        # 统计开仓收益
        temp_df = pd.DataFrame({
            'flag': flag,
            'algorithm_returns': self.algorithm_ret
        })

        temp_df['mark'] = (temp_df['flag'] != temp_df['flag'].shift(1))
        temp_df['mark'] = temp_df['mark'].cumsum()

        # 开仓至平仓的持有收益
        tradecumsumratio = temp_df.query('flag==1').groupby(
            'mark')['algorithm_returns'].sum()
        win = len(tradecumsumratio[tradecumsumratio > 0])

        wincount = round(win / open_num, 4)

        return [open_num, win, wincount]  # 交易次数，盈利次数,胜率
    
    # 画回撤百分比图
    def plot_ddpercent(self):
        
        eqiuty = self.nav_df[['algorithm_cum']]
        eqiuty['maxhere'] = eqiuty['algorithm_cum'].expanding().max()
        eqiuty['ddpercent'] = 1 - eqiuty['algorithm_cum'] / eqiuty['maxhere']
        
        eqiuty['ddpercent'].plot(figsize=(18,6),title='回撤百分比')
        end,start = self._GetMaxDrawdown(eqiuty)
        end,start = eqiuty.index[end].strftime('%Y%m%d'),eqiuty.index[start].strftime('%Y%m%d')
        
        print('平均回撤(ADD):{:.2%}'.format(eqiuty['ddpercent'].mean()))
        print('线性加权回撤(lwDD):{:.2%}'.format(talib.LINEARREG(eqiuty['ddpercent'],len(eqiuty['ddpercent']))[-1]))
        print('均方回撤(ADD^2):{:.2%}'.format(np.mean(eqiuty['ddpercent'] ** 2)))
        print(f'最大回撤开始时间范围:{start}-{end}')