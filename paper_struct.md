
Physics-consistent generative inverse design of angle-resolved Fourier-operator metasurfaces
一 Introduction
1.1 OTF engineer
1.2 逆向设计的问题：结构与角谱响应之间关系复杂、高维、非线性，经验试错效率低
1.3 已有工作：不够自由，局限于结构参数； 不通用，忽略角度 ； rcwa效率低 ；不能设计多功能
1.4 贡献：
        构建自由形态二值结构与 OTF 的数据生成/标注流程
        提出条件生成逆向设计框架： 打分---数据制备---模型训练---模型推理---样本择优的闭环
        形成通用的角度-频率-振幅三维通用设计范式：应用在二阶，偏振复用，四阶，低通，时空微分等场景SOTA


二 Methods
2.0 Framework： 简单讲+清晰框架图
2.1 dataset： 算法流程+数据增强+评价指标
2.2 forward model： 架构（简单讲）+Loss 
2.3 diffusion model： 扩散模型机制+算法流程+优势（注意力机制，物理损失）+Loss结果 
2.4 Infer： 物理引导采样说明，从数据集构造理想样本进而infer64个选最优

三 Results and Validation

3.1 高效二阶微分超表面：
* 结构图+光谱图+2D角谱图+1D-t（theta）图
* 模拟图像处理效果*2+ 指标结果分析
* 其他波长结果证明
* 和别的文章的score算一下对比一下：Score=0.988， 透过率振幅比=0.97， 工作带宽=100nm， NA=0.64


3.2 模型对比+消融实验 Comparison with baseline methods
* 代理模型对比（transformer显著提高）
* 数据scale实验
* cGAN，CVAE，diffusion w/o physics，拓扑
* 比较：Best score，Mean score，OTF MAE

3.3 Generalization to other OTF targets

* 偏振复用/偏振无关/四阶微分/时空微分结果/低通

四 Discussions （简单写）
* 为什么生成式方法适合这个问题：高维，一对多，优解稀疏，扩散比直接回归/GAN/VAE 更稳定
* 可扩展的应用场景：偏振-频率-角度三重设计

五 Conclusion
* 提出一个面向自由形态二值超表面的 OTF 逆向设计闭环。
* 在二阶微分主任务上证明其有效、稳定、可制造。
* 说明该框架对更多 OTF 工程任务有推广潜力。