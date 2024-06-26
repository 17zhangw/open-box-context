# 概览

黑盒优化（BBO）任务的目标是在有限的评估预算内对黑盒目标函数进行优化。
这里的''黑盒''指的是目标函数不具备可分析的性质，因此我们不能使用目标函数的导数等信息。
评估目标函数的代价往往是十分昂贵的。黑盒优化的目标就是尽可能快地找到一个配置，在这个配置下目标函数接近全局最优。

传统的单目标黑盒优化有很多应用场景，包括；

+ 自动化A/B测试
+ 实验设计
+ 数据库参数调优
+ **自动化超参数调优**

最近，通用的黑盒优化方法已经出现，并应用于众多领域：

+ 处理器体系结构和电路设计
+ 资源分配
+ 自动化化学设计

通用的黑盒优化方法需要支持更多的传统黑盒优化方法所不支持的功能，比如多目标优化和带约束条件的优化。

## 设计原则

OpenBox是一个高效的通用黑盒优化系统。它的设计有如下特点：

+ **易于使用**： 尽可能减少用户干预，使用用户友好的可视化界面来追踪和管理BBO任务。
+ **性能稳定**： 支持最新的优化算法。可以自动选择合适的优化算法。
+ **资源管理**： 向用户提供基于模型的使用预算的建议，例如，最小化资源预算。
+ **可扩展性**： 对输入变量的维度、目标数、任务数、测试数、以及并行度具有可扩展性。
+ **高效性**： 能够有效利用并行资源。支持基于迁移学习和多精度的优化方法。
+ **高容错性**， **支持任务广泛**， **数据隐私保护**，......


下图展示了OpenBox服务的系统概览。

<p align="center">
<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/sys_framework.svg" width="90%">
</p>


## 主要组成部分

+ **Service Master** 负责节点管理，负载均衡以及容错。

+ **Task Database** 保存所有任务的历史信息和历史状态。

+ **Suggestion Server** 给每个任务生成新的配置。

+ **REST API** 使用RESTful APIs来连接Workers和Suggestion Service。

+ **Evaluation workers** 由用户拥有和定义任务。


## 部署构件

### 单独的Python包
和其它开源包一样，OpenBox也有一个频繁的维护周期。用户可以使用Pypi或者 [GitHub](https://github.com/PKU-DAIR/open-box) 
上的源代码安装OpenBox。
我们的 [安装指南](../installation/installation_guide.md) 中提供了更多的安装细节。


### 将BBO作为一种服务
我们的目标是"把BBO作为一种服务"。我们将OpenBox实现成一个为黑盒优化提供通用的管理框架的服务。
用户可以方便地通过REStful API来访问这项服务，而无需考虑其它问题，例如环境配置，软件维护，执行优化等。
此外，OpenBox也为用户提供了Web UI来方便地追踪和管理他们运行的任务。
我们在 [部署指南](../openbox_as_service/service_deployment.md) 中提供了更多的部署细节。

## 性能对比
我们在调优LightGBM任务上使用OpenBox和六个主流的开源BBO系统进行对比。对比中我们使用25个数据集。
下图展现了比较排名（Rank值越低越好）。
我们在我们的 [论文](https://arxiv.org/abs/2106.00421) 中给出了详细的数据集描述和更多的实验结果。


<p align="center">
<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/ranking_lgb_7.svg" width="80%">
</p>

