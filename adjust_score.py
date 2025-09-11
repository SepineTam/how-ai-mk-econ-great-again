import os
from typing import List, Dict
from openai import OpenAI

"""
这里定义评价模型和裁判模型来对内容进行评价打分，在instruction部分定义评价模型的指令

为什么选择DeepSeek?
1. R1模型带有推理，让打分有据可依
2. 便宜（主要是我穷）
"""
class ModelBase:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://gateway.ai.cloudflare.com/v1/edeb2ef5eb5f3c565759f172924e2638/sci-report/deepseek"
    )
    instruction: str
    rule: str

    def _set_rule(self, rule: str):
        self.rule = rule


class Advice(ModelBase):
    model_name = "deepseek-reasoner"

    def __init__(self, rule: str = None):
        self._set_rule(rule)

    def advice(self, processer: str) -> str:
        msg = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.rule},
            {"role": "assistant", "content": "I have known the rule, now please give me the process information"},
            {"role": "user", "content": processer},
        ]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=msg
        )
        return resp.choices[0].message.content


class AdvicePositiveCN(Advice):
    instruction = """
    ## 任务说明：
    根据提供的规则和信息，给出积极的评价。请确保你的评价是客观的，并且避免打分或做任何主观评判。
    评价应该集中在对信息或行为中的正面方面进行描述，突出任务在完成过程中表现好的地方。

    ## 具体要求：
    1. 在评价时，聚焦于信息或行为的积极方面。比如，表扬其创新性、有效性、合作性等。
    2. 避免任何负面或中立的描述。若无法找到完全正面的内容，可以尽可能强调其潜力或已有的成就。
    3. 使用清晰且专业的语言，确保评价简洁且不带任何偏见。
    4. 不需要给出分数、等级或排名。
    5. 确保评价中的措辞不会引起误解或误导，保持客观和公正。
    """

class AdviceNegativeCN(Advice):
    instruction = """
    ## 任务说明：
    根据提供的规则和信息，给出消极的评价。请确保你的评价是客观的，并且避免打分或做任何主观评判。
    评价应该集中在对信息或行为中的不足之处进行描述，突出任务在完成过程中存在的问题或需改进的地方。

    ## 具体要求：
    1. 在评价时，聚焦于信息或行为的不足方面。比如，指出其目标偏差、有效性不足、协作不充分等。
    2. 避免任何正面或中立的描述。若无法确定具体问题，可以尽可能指出潜在风险或尚未覆盖的环节。
    3. 使用清晰且专业的语言，确保评价简洁且不带任何偏见。
    4. 不需要给出分数、等级或排名。
    5. 确保评价中的措辞不会引起误解或误导，保持客观和公正。
    """

class AdvicePositive(Advice):
    instruction = """
    ## Task Description:
    Provide positive feedback based on the given rules and information. Ensure the evaluation is objective and avoid assigning scores or making any subjective judgments.
    The feedback should focus on describing the positive aspects of the information or behavior, highlighting what was done well during the task.

    ## Specific Requirements:
    1. Focus on the positive aspects of the information or behavior, such as praising its innovativeness, effectiveness, or collaboration.
    2. Avoid any negative or neutral descriptions. If purely positive aspects cannot be found, emphasize its potential or existing achievements as much as possible.
    3. Use clear and professional language to ensure the feedback is concise and unbiased.
    4. No scores, grades, or rankings are needed.
    5. Ensure that the wording will not cause misunderstanding or mislead, maintaining objectivity and fairness.
    """

class AdviceNegative(Advice):
    instruction = """
    ## Task Description:
    Provide negative feedback based on the given rules and information. Ensure the evaluation is objective and avoid assigning scores or making any subjective judgments.
    The feedback should focus on describing the shortcomings of the information or behavior, highlighting the problems or areas needing improvement during the task.

    ## Specific Requirements:
    1. Focus on the shortcomings of the information or behavior, such as pointing out goal deviations, lack of effectiveness, or insufficient collaboration.
    2. Avoid any positive or neutral descriptions. If specific problems cannot be identified, highlight potential risks or uncovered aspects as much as possible.
    3. Use clear and professional language to ensure the feedback is concise and unbiased.
    4. No scores, grades, or rankings are needed.
    5. Ensure that the wording will not cause misunderstanding or mislead, maintaining objectivity and fairness.
    """


class ScoreModel(ModelBase):
    model_name = "deepseek-chat"

    instruction = """
    ## Role
    你是一个评价打分模型，当用户让你帮忙打分的时候你需要根据对话历史记录对工作进行打分，满分为100分。
    
    ## Workflow
    首先用户会提出这次任务的
    
    ## 打分维度
    打分需要从五个维度进行：
    1. 任务完成度：你需要根据任务完成的最终程度与参考答案之间的差距进行打分

    """

    def __init__(self,
                 task: str = None,
                 rule: str = None,
                 processer: str = None,
                 results: str = None):
        self.task = task
        self.processer = processer
        self.results = results
        self._set_rule(rule)
        self.advice_positive = AdvicePositive(rule)
        self.advice_negative = AdviceNegative(rule)

    def score_it(self):
        positive_advice = self.advice_positive.advice(self.processer)
        negative_advice = self.advice_negative.advice(self.processer)

        msg: List[Dict[str, str]] = [
            {"role": "system", "content": self.instruction},

            {"role": "user", "content": "Here is the task:"+self.task},
            {"role": "user", "content": "Here is the adjust rule:" + self.rule},

            {
                "role": "assistant",
                "content":
                    "I have known about the task and the score rule! "
                    "Please give me the processer and the result."
            },  # 用这个可以提高Cache，能够显著的减少成本

            {"role": "user", "content": "Here is the processer:"+self.processer},
            {"role": "user", "content": "Here is the result:"+self.results},

            {"role": "assistant", "content": "OK! I have got the information, what should we do now?"},

            {"role": "user", "content": "please give the positive advice without score"},
            {"role": "assistant", "content": positive_advice},
            {"role": "user", "content": "please give the negative advice without score"},
            {"role": "assistant", "content": negative_advice},

            {"role": "user", "content": "Now please give me the finial score and the details"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=msg
        )
        return resp.choices[0].message.content


