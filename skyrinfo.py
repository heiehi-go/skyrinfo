import micromagneticdata as md
import discretisedfield as df
import discretisedfield as df
from scipy.interpolate import CubicSpline # 三次样条差值
from scipy.optimize import brentq # 二分法找零点
import numpy as np
from scipy.fft import fft,ifft


'''
功能说明：
主要函数：
    1. outline: 提取斯格明子轮廓
    2. outline_filter: 提取滤波后的斯格明子轮廓
    3. outline_sorted: 提取斯格明子轮廓，并且对轮廓点进行排序
    4.cal_r: 计算斯格明子半径
    5.cal_center: 计算斯格明子中心



'''
def find_roots(x,y):
    '''
    x: 数组
    y: 数组
    这个函数通过x,y构造一个三次样条差值函数 f
    x应当是f的自变量,
    y应当是f的因变量。
    然后找到f的所有零点，并且以数组的形式返回

    return: root 是y = f(x)的所有零点
    '''
    # 进行插值
    f = CubicSpline(x, y)
    
    # 零点位置
    roots=[]    
    # x的步进
    step = 1e-9

    # 设置x的定义域
    min_x, max_x = x.min(), x.max()

    # 定义容差和最大迭代次数
    tolerance = 1e-12
    max_iterations = 100

    # 设置当前搜索位置为起始值
    current_x = min_x
    # 计算起始位置的函数值
    f_current = f(current_x)

    # 初始化下一个位置的数值
    next_x = current_x + step
    # 计算下一个位置的函数值
    f_next = f(next_x)

    # 遍历整个搜索区间
    while next_x < max_x:
        
        # 遍历搜索区域, 寻找左右异号的区间
        while f_current * f_next > 0: # 这意味着，零点不在当前区间
            # 步进一个位置
            current_x = next_x
            next_x += step      

            if next_x > max_x: # 搜索区间不能超过定义域
                break
            f_next = f(next_x)  # 更新函数值
            
        if next_x > max_x: # 搜索区间不能超过定义域
            break

        # 找到函数值异号的区间，使用brentq函数求解
        root = brentq(f, current_x, next_x, xtol=tolerance, maxiter=max_iterations)

        # 输出当前根
        roots.append(root)

        # 更新区间
        current_x = next_x
        f_current = f_next
        next_x += step
        f_next = f(next_x)

    return roots



def cartesian2polar(x,y):
    '''
    直角坐标转换为极坐标
    x,y都应当是数组
    '''
    if  (not isinstance(x,np.ndarray)) or (not isinstance(y,np.ndarray)):
        x = np.array(x)
        y = np.array(y)
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, r

def polar2cartesian(theta,r):
    '''
    极坐标转换为直角坐标
    theta,r: 极坐标
    '''
    if (not isinstance(theta,np.ndarray)) or (not isinstance(r,np.ndarray)):
        theta = np.array(theta)
        r = np.array(r)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y

def sort4polar(theta,r):
    '''
    极坐标排序，
    theta,r: 极坐标
    '''
    zip_r_theta = zip(r, theta)
    sorted_zip = sorted(zip_r_theta, key=lambda x:x[1])
    sorted_r, sorted_theta = zip(*sorted_zip)
    return sorted_theta,sorted_r

def band_pass_filter(signal_fft, freq, cutoff_low, cutoff_high):
    """
    带通滤波器
    :param signal_fft: 信号的FFT结果
    :param freq: 频率数组
    :param cutoff_low: 低截止频率
    :param cutoff_high: 高截止频率
    :return: 滤波后的FFT结果
    """
    # 复制FFT结果以进行操作
    filtered_fft = np.copy(signal_fft)
    # 将不在截止频率范围内的成分置零
    mask = np.logical_or(np.abs(freq) < cutoff_low, np.abs(freq) > cutoff_high)
    filtered_fft[mask] = 0
    return filtered_fft

def verify_field(obj:df.Field):
    '''
    检测输入的对象是否是一个field对象，
    同时是否满足以下要求：
    1. 以z轴为法线的平面
    2. 是一个z分量的标量
    若不满足，将他变为满足要求的field对象
    '''
    if not isinstance(obj, df.Field):  # 确保输入是一个df.Fields对象
            raise TypeError("输入对象不是df.Fields对象")
    
    if obj.nvdim != 1:
        try:
            obj = obj.z
        except AttributeError:
            raise Exception("输入对象没有z分量")
    
    if obj.mesh.region.ndim != 2:
        try:
            obj = obj.sel(z=0)
            print("警告：输入对象坐标系不是z轴为法线的平面，已自动修正为z轴为法线的平面。")
            print("z=0(手动切片可以隐藏该提示)")
        except(AttributeError, KeyError):
            raise Exception("输入对象坐标系维度异常或Z轴取不到0值")
    
    return obj


def outline(obj: df.Field, CPsign=0):
    """
    获取代表skyrmion的零点
    
    该函数通过分析给定的field对象，找出磁矩为零的点（即零点），
    这些零点用于描绘skyrmion的轮廓。函数还可以根据需要将零点的
    坐标从笛卡尔坐标系转换为极坐标系。
    
    参数:
    obj: df.Field - 一个Field对象，包含磁矩数据。
    CPsign: int - 坐标系标志，0表示保持笛卡尔坐标系，
             1表示将结果转换为极坐标系。
    
    返回:
    v1, v2: numpy.ndarray - 分别表示零点的x坐标（或半径）和y坐标（或角度）的数组。
    """
    # field对象的验证和修正
    obj = verify_field(obj)
    
    # 得到所有需要遍历的y值
    xpmin = obj.mesh.region.pmin[0]
    xpmax = obj.mesh.region.pmax[0]
    ypmin = obj.mesh.region.pmin[1]
    ypmax = obj.mesh.region.pmax[1]
    Y = np.linspace(ypmin,ypmax,obj.mesh.n[1])

    # 这是用于储存零点的两个数组
    x_zero_points = []
    y_zero_points = []

    for y in Y:
        line = obj.line(p1=(xpmin,y),p2=(xpmax,y))
        # 找到第一个和最后一个磁矩不为零的x坐标，以这种方式过滤调不想要的点
        s = line.data['v']
        first_nonzero_index = s[s != 0].index[0]
        last_nonzero_index = s[s != 0].index[-1]
        xpmin_ = line.data['x'].iloc[first_nonzero_index]
        xpmax_ = line.data['x'].iloc[last_nonzero_index]
        line_ = obj.line(p1=(xpmin_,y),p2=(xpmax_,y)) # 过滤之后的线段对象

        # 求根得到
        x = line_.data['x'].values
        value = line_.data['v'].values
        roots = find_roots(x,value)
        # 把得到的零点传到两个数组中
        for x in roots:
            x_zero_points.append(x)
            y_zero_points.append(y)
            
    x_zero_points = np.array(x_zero_points)
    y_zero_points = np.array(y_zero_points)
            
    if CPsign == 0:
        v1,v2 = x_zero_points,y_zero_points
    elif CPsign == 1:
        v1,v2 = cartesian2polar(x_zero_points,y_zero_points)
    
    return v1,v2

def outline_filter(x,y,CPsign=0):
    """
    对给定的点集进行滤波处理，以突出特定频率范围内的轮廓。

    参数:
    x -- x坐标数组
    y -- y坐标数组
    CPsign -- 坐标表示方式标志，0表示直角坐标，1表示极坐标（默认0）

    返回:
    v1 -- 经过滤波处理后的第一个坐标数组
    v2 -- 经过滤波处理后的第二个坐标数组
    """
    # 直角坐标转换成极坐标
    theta,r = cartesian2polar(x,y)

    # 进行傅里叶变换
    r_fft = fft(r)

    # 获取频率轴的值
    N = len(r)
    T = theta[1] - theta[0]  # 采样间隔
    freq = np.fft.fftfreq(N, T)

    # 应用带通滤波
    bpf_result = band_pass_filter(r_fft, freq, cutoff_low=0, cutoff_high=1)  # 假定截止频率为5Hz到10Hz
    bpf_r = np.real(ifft(bpf_result))# 对每种滤波结果进行逆傅里叶变换以回到时域
    
    # 根据CPsign的值选择不同的坐标转换方式
    if CPsign ==0: 
        # 当CPsign为0时，将极坐标转换为笛卡尔坐标
        v1,v2 = polar2cartesian(theta,bpf_r)
    elif CPsign==1:
        # 当CPsign为1时，直接使用极坐标系的值
        v1,v2 = theta,r
    return v1,v2
    
def cal_center(obj:df.Field):
    """
    计算给定对象的中心坐标。

    参数:
    obj: df.Field类型，表示一个磁矩对象，用于获取其轮廓点。

    返回值:
    x_center, y_center: 对象的中心点坐标。
    """
    x,y = outline(obj)
    x_center,y_center = x.mean(),y.mean()
    
    return x_center,y_center
    
def outline_sorted(obj:df.Field):
    '''
    计算斯格明子的轮廓点集。

    obj: df.Field对象
        需要满足的条件：
        1. 存在一个以z轴为法线的平面
        2. 存在Z分量
        
    返回值:
    x_sorted: 排序后的斯格明子轮廓点的x坐标列表
    y_sorted: 排序后的斯格明子轮廓点的y坐标列表
    '''
    
    # 1. 得到斯格明子的轮廓点集(x,y)
    x_zero_points,y_zero_points = outline(obj)
    
    # 1. 计算中心点
    x_center = np.mean(x_zero_points)
    y_center = np.mean(y_zero_points)
    
    # 1. 把所有的坐标减去中心坐标
    x_zero_points = x_zero_points - x_center
    y_zero_points = y_zero_points - y_center

    # 2. 把坐标转换为极坐标
    theta,r = cartesian2polar(x_zero_points,y_zero_points)

    # # 3. 排序
    theta_sorted,r_sorted = sort4polar(theta,r)
    
    # 7. 极坐标转换为直角坐标
    x_sorted, y_sorted = polar2cartesian(theta_sorted,r_sorted)

    # 8. 把所有坐标加上中心坐标
    x_sorted = x_sorted + x_center
    y_sorted = y_sorted + y_center
    
    
    return x_sorted, y_sorted


def cal_r(obj:df.Field,filter_sign=False):
    """
    计算斯格明子半径 r。

    参数:
    - obj: Fields 对象，表示磁场或其他矢量场的数据。
    - obj 需要满足以下要求：
        1. 存在 z 轴为法线的平面。
        2. 存在 Z 分量。

    返回:
    - R: float，计算得到的半径。
    """
    # 计算中心
    x,y=outline(obj)
    x_center,y_center = x.mean(),y.mean()
    x = x-x_center
    y = y-y_center    
    # 是否使用傅里叶滤波器
    if filter_sign:
        _,r = outline_filter(x,y,CPsign=1)
    else:
        _,r = cartesian2polar(x,y)
    
    # 最终计算半径
    if len(r)==0:
        R=0
    else:
        R = np.mean(r)
        
    return R




if __name__ == '__main__':
    import micromagneticdata as md

    data = md.Data(name="A_K",dirname="C:\\MagneticData\\EX_UNI")
    obj = data[0][0]

    print(outline(obj))
    print(outline_filter(*outline(obj)))
    print(outline_sorted(obj))
    print(cal_r(obj))
    print(cal_r(obj,filter_sign=True))