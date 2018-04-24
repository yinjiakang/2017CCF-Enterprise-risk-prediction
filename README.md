# 2017CCF-Enterprise-risk-prediction
赛题背景：
        传统的企业评价主要基于企业的财务信息，借贷记录信息等来判断企业经营状况，以及是否可能违约等信用信息。对于财务健全、在传统银行借贷领域留有记录的大中型企业，这种评价方式无疑较为客观合理。然而，对于更大量的中小微企业，既无法公开获得企业真实财务信息，也无这些企业的公开信用信息，在强变量缺失的情况下，如何利用弱变量客观公正评价企业经营状况，正是本赛题需要解决的主要问题。
        本次大赛从全国2000多万企业抽取部分企业（脱敏后），提供企业主体在多方面留下的行为足迹信息数据。参赛队伍需要通过数据挖掘的技术和机器学习的算法，针对企业未来是否会经营不善构建预测模型，输出风险预测概率值。
任务描述：
        本赛题以企业为中心，围绕企业主体在多方面留下的行为足迹信息构建训练数据集，以企业在未来两年内是否因经营不善退出市场作为目标变量进行预测。
        参赛者需要利用训练数据集中企业信息数据，构建算法模型，并利用该算法模型对验证数据集中企业，给出预测结果以及风险概率值。
评价方式：
        预测结果以AUC值作为主要评估标准，在AUC值（保留到小数点后4位数字）相同的情况下，则以F1-score（命中率及覆盖率）进行辅助评估。
赛题链接：
        http://www.datafountain.cn/#/competitions/271/intro