from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, TextElement, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from dit_model import *

# Agent画像


def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "r": 0.5
    }

    if agent.type == 5:
        portrayal["Color"] = "steelblue"
    elif agent.type == 4:
        portrayal["Color"] = "#dc9410"
    elif agent.type == 3:
        portrayal["Color"] = "#009688"
    elif agent.type == 2:
        portrayal["Color"] = "#ffeb3b"
    else:
        portrayal["Color"] = "red"

    return portrayal

# 文本信息输出


class Text(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        # 创新先驱,早期采纳者,早期大众,晚期大众,落后者
        return "innovator => " + str(model.agent_type_nums[4]) + \
            "; early adopter => " + str(model.agent_type_nums[3]) + \
            "; early majority => " + str(model.agent_type_nums[2]) + \
            "; late majority => " + str(model.agent_type_nums[1]) + \
            "; laggards => " + str(model.agent_type_nums[0])


# 实例化
text = Text()

# 折线图
chart_num_line = ChartModule(
    [
        {"Label": "agent_type1", "Color": "red"},
        {"Label": "agent_type2", "Color": "#ffeb3b"},
        {"Label": "agent_type3", "Color": "#009688"},
        {"Label": "agent_type4", "Color": "#dc9410"},
        {"Label": "agent_type5", "Color": "steelblue"},
    ]
)

chart_bs_ia_line = ChartModule(
    [
        {"Label": "model_bs_ia", "Color": "steelblue"},
        {"Label": "model_bs_ai_act", "Color": "orange"},
        {"Label": "model_bs_ai_bi", "Color": "green"},
        {"Label": "model_bs_ai_alpha", "Color": "grey"},
        {"Label": "model_ai_te", "Color": "red"},
    ]
)

# 实例化CanvasGrid()
gird = CanvasGrid(agent_portrayal, 14, 14, 500, 500)

# 模型参数:n, ei_ai_f, ei_sia, ei_srt, ei_mass, ei_tt
# 14, [0.62, 0.30, 0.50, 0.62, 0.43], 0.25, 0.50, 0.70, 0.60)
model_params = {
    "n": 14,
    "ei_ai_f": [0.62, 0.30, 0.50, 0.62, 0.43],
    "ei_sia": UserSettableParameter("slider", "school innovation atmosphere", 0.35, -1.00, 1.00, 0.01),
    "ei_srt": UserSettableParameter("slider", "school risk tolerance", 0.50, 0.00, 1.00, 0.01),
    "ei_mass": UserSettableParameter("slider", "policy advocacy", 0.50, 0.00, 1.00, 0.01),
    "ei_tt": UserSettableParameter("slider", "teacher training", 0.40, 0.00, 1.00, 0.01)
}

# 实例化服务器
server = ModularServer(
    DIT4AI,
    [text, gird, chart_num_line, chart_bs_ia_line],
    "Teachers' AI Acceptance: Agent Based Modeling",
    model_params
)

server.port = 8522
server.launch()
