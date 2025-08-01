import numpy as np

import matplotlib
import matplotlib.pyplot as plt

l1o2path = '10D2b5C0.86L53SP5Resnet1GCT100.0elr1llr1nlr001.npy'
l1expath = 'EXP5b5C0.86L53SP5Resnet1GCT100.0elr1llr1nlr001.npy'

l2o2path = '10D2b5C0.78L7SP5Resnet2GCT100.0elr-05llr1nlr001.npy'
l2expath = 'EXP5b5C0.78L7SP5Resnet2GCT100.0elr-05llr1nlr001.npy'


l1o2sen = np.load(l1o2path,allow_pickle=True)[:40,:]
l1exsen = np.load(l1expath,allow_pickle=True)[:40,:]
l2o2sen = np.load(l2o2path,allow_pickle=True)[:40,:]
l2exsen = np.load(l2expath,allow_pickle=True)[:40,:]

rounds = list(range(1, l1o2sen.shape[0] + 1))

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
a1.plot(rounds, l1o2sen[:,0], color='r', linewidth=2.5, label='Real')
a1.plot(rounds, l1o2sen[:,1], color='g', linewidth=2.5, label='Esti')

a1.set_xlabel('Round', fontdict={'size': 24}, labelpad=-1)
a1.set_ylabel('Sensitivity', fontdict={'size': 24})
a1.tick_params(labelsize=18)
a1.xaxis.set_major_locator(x_major_locator)

# 设置图例
legend = a1.legend(loc='upper right', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

#a1.set_xticks(range(0, 121, 10))
a1.set_xlim(0, 40)
a1.set_ylim(0, 830)

plt.tight_layout()
plt.savefig('Real_Esti_layers1_10D2.pdf')
plt.close(fig1)

# MLP1 EXP
fig2, a2 = plt.subplots(figsize=(6, 6))
a2.plot(rounds, l1exsen[:,0], color='r', linewidth=2.5, label='Real')
a2.plot(rounds, l1exsen[:,1], color='g', linewidth=2.5, label='Esti')

a2.set_xlabel('Round', fontdict={'size': 24}, labelpad=-1)
a2.set_ylabel('Sensitivity', fontdict={'size': 24})
a2.tick_params(labelsize=18)
a2.xaxis.set_major_locator(x_major_locator)

# 设置图例
legend = a2.legend(loc='upper right', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

#a1.set_xticks(range(0, 121, 10))
a2.set_xlim(0, 40)
a2.set_ylim(0, 830)

plt.tight_layout()
plt.savefig('Real_Esti_layers1_EXP.pdf')
plt.close(fig2)

# MLP2 10D2
fig3, a3 = plt.subplots(figsize=(6, 6))
a3.plot(rounds, l2o2sen[:,0], color='r', linewidth=2.5, label='Real')
a3.plot(rounds, l2o2sen[:,1], color='g', linewidth=2.5, label='Esti')

a3.set_xlabel('Round', fontdict={'size': 24}, labelpad=-1)
a3.set_ylabel('Sensitivity', fontdict={'size': 24})
a3.tick_params(labelsize=18)
a3.xaxis.set_major_locator(x_major_locator)

# 设置图例
legend = a3.legend(loc='upper right', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

#a1.set_xticks(range(0, 121, 10))
a3.set_xlim(0, 40)
a3.set_ylim(0, 2500)

plt.tight_layout()
plt.savefig('Real_Esti_layers2_10D2.pdf')
plt.close(fig3)

# MLP2 EXP
fig4, a4 = plt.subplots(figsize=(6, 6))
a4.plot(rounds, l2exsen[:,0], color='r', linewidth=2.5, label='Real')
a4.plot(rounds, l2exsen[:,1], color='g', linewidth=2.5, label='Esti')

a4.set_xlabel('Round', fontdict={'size': 24}, labelpad=-1)
a4.set_ylabel('Sensitivity', fontdict={'size': 24})
a4.tick_params(labelsize=18)
a4.xaxis.set_major_locator(x_major_locator)

# 设置图例
legend = a4.legend(loc='upper right', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

#a1.set_xticks(range(0, 121, 10))
a4.set_xlim(0, 40)
a4.set_ylim(0, 2500)

plt.tight_layout()
plt.savefig('Real_Esti_layers2_EXP.pdf')
plt.close(fig4)

print("所有子图已分别保存为单独的PDF文件。")