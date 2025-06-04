# PartPSP
论文《Privacy-Preserving  Decentralized Optimization via Push-SUM Protocol  with Partial Communication》中算法PartPSP的相关代码。这篇论文已投递到IEEE Transactions on Dependable and Secure Computing，正在审核中。

PartPSP是基于Push-SUM协议实现的一种具有差分隐私的有向去中心化优化算法，其主要的亮点如下：

1. 特殊的敏感度快速算法，仅需一个数字的通信即可得到整个网络的敏感度。相较于传统方法，PartPSP极大地降低了得到敏感度所需要的通信复杂度。

2. 支持部分通信，允许使用者根据模型特点有选择地通信模型组件，以此降低差分隐私所带来的模型性能衰退并提高算法的通信效率。

3. 拥有满足差分隐私的优化理论体系，其隐私保护与优化性能得到了理论证明。

技术细节与实验结果请参考论文原文，欢迎讨论与引用。
