import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行

# 导入数据：训练集和测试集
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 通过head()函数观察前5行
print(train.head(5))
print(test.head(5))

# inplace 为true则表示在原本的数据上修改，如果为False则返回新的数据
# 删除Id列因为这一列对房价结果没有影响
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# fig,ax = plt.subplots()等价于：
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# fig, ax = plt.subplots(1,3),其中参数1和3分别代表子图的行数和列数，一共有 1x3 个子图像。函数返回一个figure图像和子图ax的array列表。
# fig, ax = plt.subplots(1,3,1),最后一个参数1代表第一个子图。
# 如果想要设置子图的宽度和高度可以在函数内加入figsize值
# fig, ax = plt.subplots(1,3,figsize=(15,7))，这样就会有1行3个15x7大小的子图。

# 绘制散点图
fig, ax = plt.subplots()

# 地面以上居住面积和销售价格
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.show()

# Deleting outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
print(type(train))
print(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
# # Check the graphic again
# fig, ax = plt.subplots()
# ax.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# 画出SalePrice distribution 的KDE曲线并与正态分布作对比，也做出了QQ图（Probability Plot）与红线越接近
# 说明约符合正态分布

sns.distplot(train['SalePrice'], fit=norm)
# 获取函数使用的拟合参数
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# Now plot the distribution
# 设置图例
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
# 设置y标签
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 将数据集调整为正态分布  ？？ 为什么要将数据集调整为正态分布
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma))
plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 特征工程
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


