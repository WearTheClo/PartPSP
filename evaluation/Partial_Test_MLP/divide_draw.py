import numpy as np

import matplotlib
import matplotlib.pyplot as plt

l1o2path = './layer1/10D2b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o3path = './layer1/10D3b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o4path = './layer1/10D4b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o5path = './layer1/10D5b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o6path = './layer1/10D6b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o7path = './layer1/10D7b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o8path = './layer1/10D8b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o9path = './layer1/10D9b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l1o10path = './layer1/10D10b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'

l2o2path = './layer2/10D2b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o3path = './layer2/10D3b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o4path = './layer2/10D4b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o5path = './layer2/10D5b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o6path = './layer2/10D6b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o7path = './layer2/10D7b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o8path = './layer2/10D8b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o9path = './layer2/10D9b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l2o10path = './layer2/10D10b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'

l3o2path = './layer3/10D2b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o3path = './layer3/10D3b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o4path = './layer3/10D4b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o5path = './layer3/10D5b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o6path = './layer3/10D6b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o7path = './layer3/10D7b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o8path = './layer3/10D8b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o9path = './layer3/10D9b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'
l3o10path = './layer3/10D10b5C0.95L55SP4MLPGCT200.0elr1llr1nlr001.npy'

l1means = []
l2means = []
l3means = []

l1means.append(np.mean(np.load(l1o2path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o3path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o4path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o5path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o6path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o7path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o8path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o9path,allow_pickle=True)[:,0]))
l1means.append(np.mean(np.load(l1o10path,allow_pickle=True)[:,0]))

l2means.append(np.mean(np.load(l2o2path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o3path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o4path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o5path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o6path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o7path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o8path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o9path,allow_pickle=True)[:,0]))
l2means.append(np.mean(np.load(l2o10path,allow_pickle=True)[:,0]))

l3means.append(np.mean(np.load(l3o2path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o3path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o4path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o5path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o6path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o7path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o8path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o9path,allow_pickle=True)[:,0]))
l3means.append(np.mean(np.load(l3o10path,allow_pickle=True)[:,0]))

outdegree = list(range(2,11))


D2means = [0]
D3means = [0]
D4means = [0]
D5means = [0]
D6means = [0]
D7means = [0]
D8means = [0]
D9means = [0]
D10means = [0]

D2means.append(np.mean(np.load(l1o2path,allow_pickle=True)[:,0]))
D2means.append(np.mean(np.load(l2o2path,allow_pickle=True)[:,0]))
D2means.append(np.mean(np.load(l3o2path,allow_pickle=True)[:,0]))

D3means.append(np.mean(np.load(l1o3path,allow_pickle=True)[:,0]))
D3means.append(np.mean(np.load(l2o3path,allow_pickle=True)[:,0]))
D3means.append(np.mean(np.load(l3o3path,allow_pickle=True)[:,0]))

D4means.append(np.mean(np.load(l1o4path,allow_pickle=True)[:,0]))
D4means.append(np.mean(np.load(l2o4path,allow_pickle=True)[:,0]))
D4means.append(np.mean(np.load(l3o4path,allow_pickle=True)[:,0]))

D5means.append(np.mean(np.load(l1o5path,allow_pickle=True)[:,0]))
D5means.append(np.mean(np.load(l2o5path,allow_pickle=True)[:,0]))
D5means.append(np.mean(np.load(l3o5path,allow_pickle=True)[:,0]))

D6means.append(np.mean(np.load(l1o6path,allow_pickle=True)[:,0]))
D6means.append(np.mean(np.load(l2o6path,allow_pickle=True)[:,0]))
D6means.append(np.mean(np.load(l3o6path,allow_pickle=True)[:,0]))

D7means.append(np.mean(np.load(l1o7path,allow_pickle=True)[:,0]))
D7means.append(np.mean(np.load(l2o7path,allow_pickle=True)[:,0]))
D7means.append(np.mean(np.load(l3o7path,allow_pickle=True)[:,0]))

D8means.append(np.mean(np.load(l1o8path,allow_pickle=True)[:,0]))
D8means.append(np.mean(np.load(l2o8path,allow_pickle=True)[:,0]))
D8means.append(np.mean(np.load(l3o8path,allow_pickle=True)[:,0]))

D9means.append(np.mean(np.load(l1o9path,allow_pickle=True)[:,0]))
D9means.append(np.mean(np.load(l2o9path,allow_pickle=True)[:,0]))
D9means.append(np.mean(np.load(l3o9path,allow_pickle=True)[:,0]))

D10means.append(np.mean(np.load(l1o10path,allow_pickle=True)[:,0]))
D10means.append(np.mean(np.load(l2o10path,allow_pickle=True)[:,0]))
D10means.append(np.mean(np.load(l3o10path,allow_pickle=True)[:,0]))

layers = list(range(0, 4))


# 使用 LaTeX 排版并指定字体为 Times
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # 或其他类似字体
})

x_major_locator = plt.MultipleLocator(5)

# 创建并保存第一个子图
fig1, a1 = plt.subplots(figsize=(6, 6))
a1.plot(outdegree, l1means, color='r', linewidth=2.5, marker='o', markersize=10, label='layers=1')
a1.plot(outdegree, l2means, color='g', linewidth=2.5, marker='v', markersize=10, label='layers=2')
a1.plot(outdegree, l3means, color='b', linewidth=2.5, marker='x', markersize=10, label='layers=3')

a1.set_xlabel('d', fontdict={'size': 24}, labelpad=-1)
a1.set_ylabel('Real Average Sensitivities', fontdict={'size': 24})
a1.tick_params(labelsize=18)
a1.xaxis.set_major_locator(x_major_locator)
#a1.set_title('(a): RAS of d-Outdegree Graphs.', fontsize=24, y=-0.2)
#a1.legend(loc=0, prop={'size': 18})

# 设置图例
legend = a1.legend(loc='upper right', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

a1.set_xticks(outdegree)
a1.set_xlim(2, 10)
a1.set_ylim(0, 1500)

plt.tight_layout()
plt.savefig('RAS_outdegree.pdf')
plt.close(fig1)

# 创建并保存第二个子图
fig2, a2 = plt.subplots(figsize=(6, 6))
a2.plot(layers, D2means, color='b', linewidth=2.5, marker='o', markersize=10, label='d=2')
a2.plot(layers, D3means, color='c', linewidth=2.5, marker='o', markersize=10, label='d=3')
a2.plot(layers, D4means, color='g', linewidth=2.5, marker='o', markersize=10, label='d=4')
a2.plot(layers, D5means, color='k', linewidth=2.5, marker='o', markersize=10, label='d=5')
a2.plot(layers, D6means, color='r', linewidth=2.5, marker='o', markersize=10, label='d=6')
a2.plot(layers, D7means, color='m', linewidth=2.5, marker='o', markersize=10, label='d=7')
a2.plot(layers, D8means, color='y', linewidth=2.5, marker='o', markersize=10, label='d=8')
a2.plot(layers, D9means, color='navy', linewidth=2.5, marker='o', markersize=10, label='d=9')
a2.plot(layers, D10means, color='darkorange', linewidth=2.5, marker='o', markersize=10, label='d=10')

a2.set_xlabel('Shared Layers', fontdict={'size': 24}, labelpad=-1)
a2.set_ylabel('Real Average Sensitivities', fontdict={'size': 24})
a2.tick_params(labelsize=18)
a2.xaxis.set_major_locator(x_major_locator)
#a2.set_title('(b): Real Sensitivities of layers=1.', fontsize=24, y=-0.2)

# 设置图例
legend = a2.legend(loc='upper left', prop={'size': 18})

# 调整图例的透明度
frame = legend.get_frame()
frame.set_alpha(0.5)  # 设置整个图例框的透明度

a2.set_xticks(layers)
a2.set_xlim(0, 3)
a2.set_ylim(0, 1500)

plt.tight_layout()
plt.savefig('RAS_Layers.pdf')
plt.close(fig2)

print("所有子图已分别保存为单独的PDF文件。")