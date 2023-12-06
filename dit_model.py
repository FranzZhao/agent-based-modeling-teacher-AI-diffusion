import math

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import RandomActivation

# Agent知识库中的信息类
from dit_knowledge_class import *


# 取值范围限制在[-1,1]，同时要求数值是在创新意识附近的，不能创新意识很低，但AI行为意向却非常高，一定要有统一
def interval_limit(num, bs_ia):
    if bs_ia - 0.5 <= num <= bs_ia + 0.5:
        return max_min_1_limit(num)
    elif num > bs_ia + 0.5:
        return max_min_1_limit(bs_ia)
    elif num < bs_ia - 0.5:
        return max_min_1_limit(bs_ia)


def max_min_1_limit(num):
    # [-1, 1]
    if num > 1:
        return 1
    elif num < -1:
        return -1
    else:
        return num


def max_min_0_limit(num):
    # [0, 1]
    if num > 1:
        return 1
    elif num < 0:
        return 0
    else:
        return num


# 5个标准模型库隶属函数


def membership_fun_1(x):
    # 落后者
    if x <= -0.78:
        return 1
    elif -0.78 < x < -0.68:
        return (-0.68 - x) / 0.1
    else:
        return 0


def membership_fun_2(x):
    # 晚期大众
    if x <= -0.78:
        return 0
    elif -0.78 < x < -0.68:
        return (x + 0.75) / 0.1
    elif -0.68 <= x <= -0.10:
        return 1
    elif -0.10 < x < 0.00:
        return - x / 0.1
    else:
        return 0


def membership_fun_3(x):
    # 早期大众
    if x <= -0.10:
        return 0
    elif -0.10 < x < 0.00:
        return (x + 0.10) / 0.10
    elif 0.10 <= x <= 0.58:
        return 1
    elif 0.58 < x < 0.68:
        return (0.68 - x) / 0.1
    else:
        return 0


def membership_fun_4(x):
    # 早期采纳者
    if x <= 0.58:
        return 0
    elif 0.58 < x < 0.68:
        return (x - 0.85) / 0.1
    elif 0.68 <= x <= 0.85:
        return 1
    elif 0.85 < x < 0.95:
        return (0.95 - x) / 0.1
    else:
        return 0


def membership_fun_5(x):
    # 创新先驱
    if x <= 0.85:
        return 0
    elif 0.85 < x < 0.95:
        return (x - 0.85) / 0.1
    else:
        return 1


# 模糊模型识别
def fuzzy_type_recognize(agent):
    a1 = membership_fun_1(agent.bs.bs_ia)
    a2 = membership_fun_2(agent.bs.bs_ia)
    a3 = membership_fun_3(agent.bs.bs_ia)
    a4 = membership_fun_4(agent.bs.bs_ia)
    a5 = membership_fun_5(agent.bs.bs_ia)
    a = [a1, a2, a3, a4, a5]
    agent.type = a.index(max(a)) + 1


# Rasch模型
def rasch_model(x, a, b, c):
    # 公式：y = 1/(1+e^(ax+b)) + c
    rasch_e = math.e ** (a * x + b)
    rasch_val = 1 / (1 + rasch_e) + c
    return rasch_val


# 模型输出结果，平均创新意识


def model_final_bs_ia(model):
    return model.model_bs_ia


def model_final_bs_ai_act(model):
    return model.model_bs_ai_act


def model_final_bs_ai_bi(model):
    return model.model_bs_ai_bi


def model_final_bs_ai_alpha(model):
    return model.model_bs_ai_alpha


def model_final_ai_te(model):
    return model.model_ai_te


class Teacher(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 0  # 创新采纳者类型,[1,2,3,4,5]
        self.ai_te = 0  # AI教学成效
        self.bs = BeliefSpace()  # 信念空间
        self.ks = KnowledgeSpace()  # 知识空间
        self.ei = EnvInfo()  # 外界信息
        self.bs_ai_act_kh = 0
        self.bs_ai_bi_kh = 0

    # 步骤1：认知阶段-获取外界信息
    def get_environment_information(self):
        # 获取的信息包括：创新意识bs_ia, 感知易用性bs_eou, 感知有用性bs_usf, 知识空间中的ks_ak, ks_htk, ks_pk
        self.ei.ei_ir = []
        # 获取整个系统的Agent
        cells = self.model.grid.coord_iter()
        for cell in cells:
            cell_content, x, y = cell
            self.ei.ei_ir.append({
                "type": cell_content.type,
                "bs_ia": cell_content.bs.bs_ia,
                "bs_eou": cell_content.bs.bs_eou,
                "bs_usf": cell_content.bs.bs_usf,
                "ks_ak": cell_content.ks.ks_ak,
                "ks_htk": cell_content.ks.ks_htk,
                "ks_pk": cell_content.ks.ks_pk,
                "bs_ai_act": cell_content.bs.bs_ai_act,
                "bs_ai_bi": cell_content.bs.bs_ai_bi
            })

    # 步骤2：说服阶段-信念空间与知识空间的更新
    def renew_bs_ks(self):
        # 说服变量：创新意识bs_ia,感知易用性bs_eou,感知有用性bs_usf,知晓性ks_ak,程序性ks_htk,原理性ks_pk
        bs_ia_up = 0
        bs_ia_down = 0
        bs_eou_up = 0
        bs_eou_down = 0
        bs_usf_up = 0
        bs_usf_down = 0
        ks_ak_up = 0
        ks_ak_down = 0
        ks_htk_up = 0
        ks_htk_down = 0
        ks_pk_up = 0
        ks_pk_down = 0
        # 决策变量：AI接纳度bs_ai_act,AI行为意向bs_ai_bi
        bs_ai_act_up = 0
        bs_ai_act_down = 0
        bs_ai_bi_up = 0
        bs_ai_bi_down = 0
        # 分析所有agent
        for i in range(len(self.ei.ei_ir)):
            # 观点差异计算
            diff_bs_ia = self.bs.bs_ia - self.ei.ei_ir[i]["bs_ia"]
            diff_bs_ee = self.bs.bs_eou - self.ei.ei_ir[i]["bs_eou"]
            diff_bs_pe = self.bs.bs_usf - self.ei.ei_ir[i]["bs_usf"]
            diff_ks_ak = self.ks.ks_ak - self.ei.ei_ir[i]["ks_ak"]
            diff_ks_htk = self.ks.ks_htk - self.ei.ei_ir[i]["ks_htk"]
            diff_ks_pk = self.ks.ks_pk - self.ei.ei_ir[i]["ks_pk"]
            diff_bs_ai_act = self.bs.bs_ai_act - self.ei.ei_ir[i]["bs_ai_act"]
            diff_bs_ai_bi = self.bs.bs_ai_bi - self.ei.ei_ir[i]["bs_ai_bi"]

            mu = 0.15
            # 创新意识bs_ia
            if math.fabs(diff_bs_ia) < mu:
                bs_ia_w = self.ei.ei_ir[i]["type"] * \
                          rasch_model(diff_bs_ia, 20, -4, 0)
                bs_ia_up += bs_ia_w * diff_bs_ia
                bs_ia_down += bs_ia_w
            # 努力期望bs_ee
            if math.fabs(diff_bs_ee) < mu:
                bs_ee_w = self.ei.ei_ir[i]["type"] * \
                          rasch_model(diff_bs_ee, 20, -4, 0)
                bs_eou_up += bs_ee_w * diff_bs_ee
                bs_eou_down += bs_ee_w
            # 表现期望bs_pe
            if math.fabs(diff_bs_pe) < mu:
                bs_pe_w = self.ei.ei_ir[i]["type"] * \
                          rasch_model(diff_bs_ee, 20, -4, 0)
                bs_usf_up += bs_pe_w * diff_bs_pe
                bs_usf_down += bs_pe_w
            # 知晓性知识ks_ak
            if math.fabs(diff_ks_ak) < mu:
                ks_ak_w = self.ei.ei_ir[i]["type"] * \
                          rasch_model(diff_ks_ak, 20, -4, 0)
                ks_ak_up += ks_ak_w * diff_ks_ak
                ks_ak_down += ks_ak_w
            # 程序性知识ks_htk
            if math.fabs(diff_ks_htk) < mu:
                ks_htk_w = self.ei.ei_ir[i]["type"] * \
                           rasch_model(diff_ks_htk, 20, -4, 0)
                ks_htk_up += ks_htk_w * diff_ks_htk
                ks_htk_down += ks_htk_w
            # 原理性知识ks_pk
            if math.fabs(diff_ks_pk) < mu:
                ks_pk_w = self.ei.ei_ir[i]["type"] * \
                          rasch_model(diff_ks_pk, 20, -4, 0)
                ks_pk_up += ks_pk_w * diff_ks_pk
                ks_pk_down += ks_pk_w
            # AI接纳度bs_ai_act
            if math.fabs(diff_bs_ai_act) < mu:
                bs_ai_act_w = self.ei.ei_ir[i]["type"] * \
                              rasch_model(diff_bs_ai_act, 20, -4, 0)
                bs_ai_act_up += bs_ai_act_w * diff_bs_ai_act
                bs_ai_act_down += bs_ai_act_w
            # AI行为意向bs_ai_bi
            if math.fabs(diff_bs_ai_bi) < mu:
                bs_ai_bi_w = self.ei.ei_ir[i]["type"] * \
                             rasch_model(diff_bs_ai_bi, 20, -4, 0)
                bs_ai_bi_up += bs_ai_bi_w * diff_bs_ai_bi
                bs_ai_bi_down += bs_ai_bi_w

        # 风险容忍度
        rho_ei_srt = rasch_model(self.ei.ei_srt, 10, -5, 0.5)
        # 创新意识bs_ia更新
        self.bs.bs_ia += (bs_ia_up / bs_ia_down + 0.06 *
                          self.ei.ei_sia) / rho_ei_srt
        self.bs.bs_ia = max_min_1_limit(self.bs.bs_ia)
        fuzzy_type_recognize(self)

        # 感知易用性bs_eou更新
        w_f_eou = [0.2, 0.5, 0.5, 0.2, 0.2]
        self.bs.bs_eou += 0.03 * (bs_eou_up / bs_eou_down) + 0.07 * np.dot(w_f_eou, self.ei.ei_ai_f) / 5
        self.bs.bs_eou = interval_limit(self.bs.bs_eou, self.bs.bs_ia)

        # 感知有用性bs_usf更新
        w_f_usf = [0.5, 0.2, 0.2, 0.5, 0.5]
        self.bs.bs_usf += 0.04 * self.bs.bs_eou + 0.03 * (bs_usf_up / bs_usf_down) + \
                          0.07 * np.dot(w_f_usf, self.ei.ei_ai_f) / 5
        self.bs.bs_usf = interval_limit(self.bs.bs_usf, self.bs.bs_ia)

        # 知晓性知识ks_ak更新
        self.ks.ks_ak += 0.06 * self.ei.ei_mass + \
                         0.04 * (ks_ak_up / ks_ak_down)
        self.ks.ks_ak = max_min_0_limit(self.ks.ks_ak)

        # 程序性知识ks_htk更新
        self.ks.ks_htk += 0.07 * self.ei.ei_tt + \
                          0.03 * (ks_htk_up / ks_htk_down)
        self.ks.ks_htk = max_min_0_limit(self.ks.ks_htk)

        # 原理知识ks_pk更新
        self.ks.ks_pk += 0.08 * self.ei.ei_tt + 0.02 * (ks_pk_up / ks_pk_down)
        self.ks.ks_pk = max_min_0_limit(self.ks.ks_pk)

        self.bs_ai_act_kh = bs_ai_act_up / bs_ai_act_down
        self.bs_ai_bi_kh = bs_ai_bi_up / bs_ai_bi_down

    # 步骤3：决策阶段-AI采纳程度确定
    def decide_ai_act(self):
        # 新算法：K-H模型更新
        # AI接纳度更新
        self.bs.bs_ai_act += 0.02 * self.bs.bs_ia + 0.03 * self.bs.bs_usf + 0.03 * self.bs.bs_eou + \
                             0.04 * (self.ks.ks_ak + self.ks.ks_htk + self.ks.ks_pk) / \
                             3 + 0.06 * self.bs_ai_act_kh
        # AI行为意向更新
        self.bs.bs_ai_bi += 0.02 * self.bs.bs_ia + 0.03 * self.bs.bs_usf + 0.04 * self.bs.bs_ai_act + \
                            0.03 * (self.ks.ks_ak + self.ks.ks_htk + self.ks.ks_pk) / \
                            3 + 0.06 * self.bs_ai_bi_kh

        # 区间范围限制
        self.bs.bs_ai_act = interval_limit(self.bs.bs_ai_act, self.bs.bs_ia)
        self.bs.bs_ai_bi = interval_limit(self.bs.bs_ai_bi, self.bs.bs_ia)

    # 步骤4：执行阶段-AI教学实践成效
    def ai_teaching_practice(self):
        # AI教学风险水平bs_alpha
        ks_val = 0.2 * self.ks.ks_ak + 0.3 * self.ks.ks_htk + 0.5 * self.ks.ks_pk
        ks_risk_level = rasch_model(ks_val, 8, -4, 0)  # 知识水平风险
        ai_f_val = sum(self.ei.ei_ai_f) / 5
        ei_ai_f_risk_level = rasch_model(ai_f_val, 3, 0, 0)  # AI技术属性风险
        risk_level = 0.5 * ks_risk_level + 0.5 * ei_ai_f_risk_level
        self.bs.bs_alpha = rasch_model(
            risk_level, -5, self.ei.ei_sia + 2.5, 0) - 0.8 * self.bs.bs_ia
        if self.bs.bs_alpha > 1:
            self.bs.bs_alpha = 1
        elif self.bs.bs_alpha < 0:
            self.bs.bs_alpha = 0
        self.model.model_bs_ai_alpha += self.bs.bs_alpha

        # AI教学风险水平bs_alpha与学校风险容忍度ei_srt的差异判定
        diff = self.bs.bs_alpha - self.ei.ei_srt
        beta = rasch_model(diff, -4, 0, 0)

        # AI教学成效ai_te model_ai_te
        rasch_x = 0.7 * self.bs.bs_ai_bi + 0.3 * self.ei.ei_sia
        self.ai_te = max_min_1_limit(2 * rasch_model(rasch_x, -4, 2 * beta, 0) - 1)

    # 步骤5：确认阶段-AI接纳度调整
    def modify_ai_act(self):
        # 基于AI教学成效的调节幅度
        mu = (1 / (3 + math.e ** (1 - 5 * self.ai_te)) - 0.175) * 0.1
        # 创新意识bs_ia
        self.bs.bs_ia += mu
        self.bs.bs_ia = max_min_1_limit(self.bs.bs_ia)
        # 创新采纳类型判断
        fuzzy_type_recognize(self)
        # AI接纳度bs_ai_act
        self.bs.bs_ai_act += mu
        self.bs.bs_ai_act = interval_limit(self.bs.bs_ai_act, self.bs.bs_ia)
        # AI行为意向bs_ai_bi
        self.bs.bs_ai_bi += mu
        self.bs.bs_ai_bi = interval_limit(self.bs.bs_ai_bi, self.bs.bs_ia)

    def step(self):
        # 步骤1：认知阶段
        self.get_environment_information()
        # 步骤2：说服阶段
        self.renew_bs_ks()
        # 步骤3：决策阶段
        self.decide_ai_act()
        # 步骤4：执行阶段
        self.ai_teaching_practice()
        # 步骤5：确认阶段
        self.modify_ai_act()
        # 模型数据收集
        self.model.agent_type_nums[self.type - 1] += 1
        self.model.model_bs_ia += self.bs.bs_ia
        self.model.model_bs_ai_act += self.bs.bs_ai_act
        self.model.model_bs_ai_bi += self.bs.bs_ai_bi
        self.model.model_ai_te += self.ai_te


class DIT4AI(Model):

    def __init__(self, n, ei_ai_f, ei_sia, ei_srt, ei_mass, ei_tt):
        """
        :param n: Agent数量
        :param ei_ai_f: AI技术属性特征[f1,f2,f3,f4,f5]
        :param ei_sia: 学校创新氛围
        :param ei_srt: 学校风险容忍度
        :param ei_mass: 大众渠道宣传
        :param ei_tt: 教师培训
        """
        self.num_agents = n * n
        # 个数统计：落后者、晚期大众、早期大众、早期采纳者、创新先驱
        self.agent_type_nums = [0, 0, 0, 0, 0]
        self.agent_type1 = 0
        self.agent_type2 = 0
        self.agent_type3 = 0
        self.agent_type4 = 0
        self.agent_type5 = 0
        # 模型外界信息EI初始化
        self.ei_ai_f = ei_ai_f
        self.ei_sia = ei_sia
        self.ei_srt = ei_srt
        self.ei_mass = ei_mass
        self.ei_tt = ei_tt
        # 模型中的Agent总体参数
        self.model_bs_ia = 0  # 创新意识BS_IA
        self.model_bs_ai_act = 0  # AI接纳度BS_AI_ACT
        self.model_bs_ai_bi = 0  # AI教学意向BS_AI_BI
        self.model_bs_ai_alpha = 0  # AI教学风险BS_AI_alpha
        self.model_ai_te = 0  # AI教学成效AI_TE
        # 模型相关
        self.grid = SingleGrid(n, n, True)
        self.schedule = RandomActivation(self)
        self.running = True

        # 创新意识正态随机数生成，并限制其在[-1,1]之间
        bs_ia_array = np.random.normal(0, 0.4, self.num_agents)
        for i in range(len(bs_ia_array)):
            bs_ia_array[i] = max_min_1_limit(bs_ia_array[i])

        # Agent初始化
        for i in range(self.num_agents):
            # Agent实例化
            new_agent = Teacher(i + 1, self)
            # Agent参数初始化
            new_agent.bs.bs_ia = bs_ia_array[i]
            # Agent类型判断：type
            fuzzy_type_recognize(new_agent)
            # Agent教学成效初始化0
            new_agent.ai_te = 0
            # Agent的AI教学风险水平初始化0
            new_agent.bs.bs_alpha = 0
            # Agent信念空间&知识空间：围绕bs_ia的7个正态随机数
            random_val = np.random.normal(new_agent.bs.bs_ia, 0.03, 7)
            for j in range(7):
                random_val[j] = interval_limit(
                    random_val[j], new_agent.bs.bs_ia)
            new_agent.bs.bs_eou = random_val[0]  # 感知易用性
            new_agent.bs.bs_usf = random_val[1]  # 感知有用性
            new_agent.bs.bs_ai_act = random_val[2]  # AI接纳度
            new_agent.bs.bs_ai_bi = random_val[3]  # AI行为意向
            new_agent.ks.ks_ak = random_val[4]  # 知晓性知识
            new_agent.ks.ks_htk = random_val[5]  # 程序性知识
            new_agent.ks.ks_pk = random_val[6]  # 原理性知识
            # Agent外界信息
            new_agent.ei.ei_ai_f = self.ei_ai_f
            new_agent.ei.ei_sia = self.ei_sia
            new_agent.ei.ei_srt = self.ei_srt
            new_agent.ei.ei_mass = self.ei_mass
            new_agent.ei.ei_tt = self.ei_tt

            # 模型信息统计
            self.agent_type_nums[new_agent.type - 1] += 1
            self.model_bs_ia += new_agent.bs.bs_ia
            self.model_bs_ai_act += new_agent.bs.bs_ai_act
            self.model_bs_ai_bi += new_agent.bs.bs_ia
            self.model_bs_ai_alpha += new_agent.bs.bs_alpha
            self.model_ai_te += new_agent.ai_te

            # 放置Agent
            while True:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                if self.grid.is_cell_empty((x, y)):
                    self.grid.place_agent(new_agent, (x, y))
                    break

            # 将实例化后的Agent加入Schedule中
            self.schedule.add(new_agent)

        # 数据处理
        self.data_process()
        # 收集输出
        self.datacollector = DataCollector(
            model_reporters={
                "agent_type1": "agent_type1",
                "agent_type2": "agent_type2",
                "agent_type3": "agent_type3",
                "agent_type4": "agent_type4",
                "agent_type5": "agent_type5",
                "model_bs_ia": model_final_bs_ia,
                "model_bs_ai_act": model_final_bs_ai_act,
                "model_bs_ai_bi": model_final_bs_ai_bi,
                "model_bs_ai_alpha": model_final_bs_ai_alpha,
                "model_ai_te": model_final_ai_te
            }
        )

    def step(self):
        self.agent_type_nums = [0, 0, 0, 0, 0]
        self.model_bs_ia = 0
        self.model_bs_ai_act = 0
        self.model_bs_ai_bi = 0
        self.model_bs_ai_alpha = 0
        self.model_ai_te = 0
        self.schedule.step()
        # 数据处理
        self.data_process()
        self.datacollector.collect(self)

    def data_process(self):
        self.agent_type1 = self.agent_type_nums[0]
        self.agent_type2 = self.agent_type_nums[1]
        self.agent_type3 = self.agent_type_nums[2]
        self.agent_type4 = self.agent_type_nums[3]
        self.agent_type5 = self.agent_type_nums[4]
        self.model_bs_ia /= self.num_agents
        self.model_bs_ai_act /= self.num_agents
        self.model_bs_ai_bi /= self.num_agents
        self.model_bs_ai_alpha /= self.num_agents
        self.model_ai_te /= self.num_agents

# if __name__ == "__main__":
#     model = DIT4AI(14, [0.62, 0.30, 0.50, 0.62, 0.43], 0.25, 0.50, 0.70, 0.60)
