import math
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.integrate import quad # 计算定积分
from scipy.stats import norm 
mpl.rcParams['font.family'] = 'serif' # matplotlib显示中文方法

# BSM模型定价
def BSM_call_value(St, K, t, T, r, sigma):
    '''
    Calculate BSM European call option value. 
    Parameters
    ==========
    St: float
        stock/index level at time t 
    K: float
        strike price
    t: float
        valuation date
    T: float
        date of maturity/time-to-maturity if t = 0; T > t
    r: float
        constant, risk-free short rate
    sigma: float
        volatility
    Returns
    ============
    call_value: float
        European call present value at t
    '''

    d1 = (np.log(St/K) + (r + 0.5 * sigma ** 2) * (T - t))/(sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    call_value = St * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    return call_value

def BSM_put_value(St, K, t, T, r, sigma):
    '''
    Calculate BSM European put option value. 
    Parameters
    ==========
    St: float
        stock/index level at time t 
    K: float
        strike price
    t: float
        valuation date
    T: float
        date of maturity/time-to-maturity if t = 0; T > t
    r: float
        constant, risk-free short rate
    sigma: float
        volatility
    Returns
    ============
    put_value: float
        European put present value at t
    '''
    put_value = BSM_call_value(St, K, t, T, r, sigma) - St + np.exp(-r * (T - t)) * K
    return put_value

# CRR欧式期权价格
def CRR_european_option_value(S0, K, T, r, sigma, otype, M=4):
    '''Cox-Ross-Rubinstein European option valuation.
    Parameters
    ==========
    S0: float
        stock/index level at time 0
    K: float
        strike price
    T: float
        date of maturity
    r: float
        constant, riskfree short rate
    sigma: float
        volatility
    otype: string
        either 'call' or 'put'
    M: int
        number of time intervals
    '''
    # 生成二叉树
    dt = T / M # length of time intervals
    df = math.exp(-r * dt) # discount per interval 

    # 计算u、d、q
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u 
    q = (math.exp(r * dt) - d) / (u - d) 

    # 初始化幂矩阵
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = mu.T

    mu = u ** (mu - md) # mu - md相当于是所有的二叉树路径了
    md = d ** md 
    S = S0 * mu * md # 得到各节点的股票价值

    # 得到叶子节点的期权价值
    if otype == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # 逐步向前加权平均并折现，得到期初期权价值
    for z in range(0, M): 
        V[0: M - z, M - z - 1] = (q * V[0:M - z, M - z]+ (1 - q) * V[1:M -z + 1, M - z]) * df
        
    return V[0, 0]

# CRR美式期权
def CRR_american_option_value(S0, K, T, r, sigma, otype, M=4):
    # 一、生成二叉树
    dt = T / M
    df = math.exp(-r * dt)
    inf = math.exp(r * dt)

    # 计算u、d、p
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    # 初始化幂矩阵
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = mu.T

    # 股票价格
    S = S0 * u ** (mu - md) * d ** md
    
    # 二、计算各节点的股票价格
    mes = S0 * inf ** mu

    # 三、得到叶子节点的期权价值
    if otype == 'call': 
        V = np.maximum(S - K, 0)
        oreturn = mes - K
    else:
        V = np.maximum(K - S, 0)
        oreturn = K - mes

    # 四、逐步向前加权平均折现和提前行权的收益比较，得到期初期权价值
    for z in range(M):
        ovalue = (q * V[0:M - z, M - z]) + ((1 - q) * V[1:M -z + 1, M - z]) * df
        V[0: M - z, M - z - 1] = np.maximum(ovalue, oreturn[0: M - z, M - z - 1])
    
    return V[0, 0]

# 输入参数
S0 = 100.00 # index level
K = 100.00 # option strike
T = 1.00 # maturity date
r = 0.05 # riskfree rate
sigma = 0.20 # volatility
mmin = 2
mmax = 200
step_size = 1

print(CRR_american_option_value(S0, K, T, r, sigma, 'call', 100))
print(CRR_european_option_value(S0, K, T, r, sigma, 'call', 100))

BSM_benchmark = BSM_call_value(S0, K, 0, T, r, sigma)
m = range(mmin, mmax, step_size)
CRR_values = [CRR_european_option_value(S0, K, T, r, sigma, 'call', M) for M in m]
plt.figure(figsize=(9, 5))
plt.plot(m, CRR_values, label='CRR')
plt.axhline(BSM_benchmark, color='r', ls='dashed', lw=1.5, label='BSM')
plt.xlabel('Steps')
plt.ylabel('European call option value')
plt.legend(loc=4)
plt.xlim(0, mmax)
plt.show()


