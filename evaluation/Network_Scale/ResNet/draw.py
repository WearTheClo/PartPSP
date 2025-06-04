import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n10depath = '10D1b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n10d2path = '10D2b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n10d4path = '10D4b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n10d6path = '10D6b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n10d8path = '10D8b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'

n20depath = '20D1b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n20d2path = '20D2b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n20d4path = '20D4b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n20d6path = '20D6b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n20d8path = '20D8b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'

n30depath = '30D1b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n30d2path = '30D2b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n30d4path = '30D4b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n30d6path = '30D6b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n30d8path = '30D8b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'

n40depath = '40D1b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n40d2path = '40D2b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n40d4path = '40D4b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n40d6path = '40D6b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n40d8path = '40D8b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'

n50depath = '50D1b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n50d2path = '50D2b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n50d4path = '50D4b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n50d6path = '50D6b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'
n50d8path = '50D8b5C0.78L7SP4Resnet18GCT100.0elr-05llr1nlr001.npy'

n10demean = np.mean(np.load(n10depath,allow_pickle=True)[:20,0])
n10d2mean = np.mean(np.load(n10d2path,allow_pickle=True)[:20,0])
n10d4mean = np.mean(np.load(n10d4path,allow_pickle=True)[:20,0])
n10d6mean = np.mean(np.load(n10d6path,allow_pickle=True)[:20,0])
n10d8mean = np.mean(np.load(n10d8path,allow_pickle=True)[:20,0])

n20demean = np.mean(np.load(n20depath,allow_pickle=True)[:20,0])
n20d2mean = np.mean(np.load(n20d2path,allow_pickle=True)[:20,0])
n20d4mean = np.mean(np.load(n20d4path,allow_pickle=True)[:20,0])
n20d6mean = np.mean(np.load(n20d6path,allow_pickle=True)[:20,0])
n20d8mean = np.mean(np.load(n20d8path,allow_pickle=True)[:20,0])

n30demean = np.mean(np.load(n30depath,allow_pickle=True)[:20,0])
n30d2mean = np.mean(np.load(n30d2path,allow_pickle=True)[:20,0])
n30d4mean = np.mean(np.load(n30d4path,allow_pickle=True)[:20,0])
n30d6mean = np.mean(np.load(n30d6path,allow_pickle=True)[:20,0])
n30d8mean = np.mean(np.load(n30d8path,allow_pickle=True)[:20,0])

n40demean = np.mean(np.load(n40depath,allow_pickle=True)[:20,0])
n40d2mean = np.mean(np.load(n40d2path,allow_pickle=True)[:20,0])
n40d4mean = np.mean(np.load(n40d4path,allow_pickle=True)[:20,0])
n40d6mean = np.mean(np.load(n40d6path,allow_pickle=True)[:20,0])
n40d8mean = np.mean(np.load(n40d8path,allow_pickle=True)[:20,0])

n50demean = np.mean(np.load(n50depath,allow_pickle=True)[:20,0])
n50d2mean = np.mean(np.load(n50d2path,allow_pickle=True)[:20,0])
n50d4mean = np.mean(np.load(n50d4path,allow_pickle=True)[:20,0])
n50d6mean = np.mean(np.load(n50d6path,allow_pickle=True)[:20,0])
n50d8mean = np.mean(np.load(n50d8path,allow_pickle=True)[:20,0])

delist = [n10demean, n20demean, n30demean, n40demean, n50demean]
d2list = [n10d2mean, n20d2mean, n30d2mean, n40d2mean, n50d2mean]
d4list = [n10d4mean, n20d4mean, n30d4mean, n40d4mean, n50d4mean]
d6list = [n10d6mean, n20d6mean, n30d6mean, n40d6mean, n50d6mean]
d8list = [n10d8mean, n20d8mean, n30d8mean, n40d8mean, n50d8mean]

N = [10, 20, 30, 40, 50]

# 使用 LaTeX 排版并指定字体为 Times
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # 或其他类似字体
})

# 添加 LaTeX 前置命令以加载 Times 字体包
#plt.rcParams['text.latex.preamble'] = r'\usepackage{newtxtext}'

x_major_locator = plt.MultipleLocator(5)

# MLP1 10D2
fig1, a1 = plt.subplots(figsize=(6, 6))
a1.plot(N, delist, color='r', linewidth=2.5, marker='x', markersize=10, label='Exp')
a1.plot(N, d2list, color='g', linewidth=2.5, marker='+', markersize=10, label='d=2')
a1.plot(N, d4list, color='b', linewidth=2.5, marker='o', markersize=10, label='d=4')
a1.plot(N, d6list, color='k', linewidth=2.5, marker='>', markersize=10, label='d=6')
a1.plot(N, d8list, color='c', linewidth=2.5, marker='<', markersize=10, label='d=8')

a1.set_xlabel('Network Scale N', fontdict={'size': 24}, labelpad=-1)
a1.set_ylabel('Real Average Sensitivities', fontdict={'size': 24})
a1.tick_params(labelsize=18)
a1.xaxis.set_major_locator(x_major_locator)

# 设置图例
legend = a1.legend(loc='lower right', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

a1.set_xticks(range(10, 51, 10))
a1.set_xlim(10, 50)
a1.set_ylim(0, 20)

a1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.tight_layout()
plt.savefig('RAS_N_Res.pdf')
plt.close(fig1)

print("所有子图已分别保存为单独的PDF文件。")