class BeliefSpace:
    def __init__(self, bs_ia=0, bs_eou=0, bs_usf=0, bs_ai_act=0, bs_ai_bi=0, bs_alpha=0):
        """
        :param bs_ia: 创新意识
        :param bs_eou: 感知易用性
        :param bs_usf: 感知有用性
        :param bs_ai_act: AI接纳度
        :param bs_ai_bi: AI行为意向
        :param bs_alpha: AI教学风险水平
        """
        self.bs_ia = bs_ia
        self.bs_eou = bs_eou
        self.bs_usf = bs_usf
        self.bs_ai_act = bs_ai_act
        self.bs_ai_bi = bs_ai_bi
        self.bs_alpha = bs_alpha


class KnowledgeSpace:
    def __init__(self, ks_ak=0, ks_htk=0, ks_pk=0):
        """
        :param ks_ak: 知晓性知识
        :param ks_htk: 程序性知识
        :param ks_pk: 原理性知识
        """
        self.ks_ak = ks_ak
        self.ks_htk = ks_htk
        self.ks_pk = ks_pk


class EnvInfo:
    def __init__(self, ei_ai_f=0, ei_sia=0, ei_srt=0, ei_mass=0, ei_tt=0, ei_ir=0):
        """
        :param ei_ai_f: AI技术属性特征
        :param ei_sia: 学校创新氛围
        :param ei_srt: 学校风险容忍度
        :param ei_mass: 大众渠道宣传
        :param ei_tt: 教师培训
        :param ei_ir: 人际关系
        """
        self.ei_ai_f = ei_ai_f
        self.ei_sia = ei_sia
        self.ei_srt = ei_srt
        self.ei_mass = ei_mass
        self.ei_tt = ei_tt
        self.ei_ir = ei_ir
