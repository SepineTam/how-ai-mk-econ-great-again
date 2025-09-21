#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : extract_info.py

"""
This file is based on the Google's LangExtract.
If you are finding more powerful usage method, you can get it from this repo.
"""
import os

import langextract as lx
import textwrap
# If you are located in China, you also could use DeepSeek as your model provider.
from langextract.providers.openai import OpenAILanguageModel


class ScoreExtract:
    # Set model provider
    # You can change it to any model provider whatever you want.
    model = OpenAILanguageModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("LEARN_BASE_URL"),
        model_id="deepseek-chat"
    )

    # 1. Define the prompt and extraction rules
    prompt_text = "仅提取评分信息：总分、结果分（及原因）、过程分（及原因）、四个维度分（及原因）。不要改写原文。"

    prompt = textwrap.dedent(prompt_text)


    # 2. Provide a high-quality example to guide the model
    example_text = """
    <meta>
    TaskID: C2_C1
    Language: EN
    </meta>
    <score>
    <score_from_results>
    Result Score: 0 / 20
    Reason: The user's work failed to produce any empirical results due to the inability to access the SLEEP75 dataset. No estimation, variance comparison, or hypothesis testing was performed, rendering the task incomplete. The theoretical explanation provided does not substitute for the required data-driven analysis, and no actual numerical or statistical outputs were generated.
    </score_from_results>
    <score_from_processer>
    Processer Score: 32 / 80
    Reason: While the Agent showed structured reasoning and attempted tool usage, the overall process was hampered by critical failures in error handling and data accessibility, leading to an incomplete task execution.
    <detail>
    <agent_reasoning>
    Score: 15 / 20
    Reason: The Agent correctly understood the task requirements and outlined a logical approach for heteroskedasticity analysis, including model specification, estimation steps, and hypothesis testing. However, reasoning was not fully adaptive to the unavailability of data.
    </agent_reasoning>
    <tool_usage>
    Score: 10 / 20
    Reason: Tools were actively used to attempt data loading and Stata script execution, but repeated failures to locate the dataset were not adequately resolved. No alternative strategies (e.g., data simulation, manual input) were employed.
    </tool_usage>
    <error_handling>
    Score: 2 / 20
    Reason: The Agent did not effectively handle the missing dataset error. Instead of pivoting to a feasible solution (e.g., using built-in example data, requesting user clarification, or simulating data), it persisted with unavailable resources and ultimately provided only theoretical output.
    </error_handling>
    <planning>
    Score: 5 / 20
    Reason: Initial steps were organized, but the plan lacked contingency measures for data unavailability. The process was not resilient, and the final output did not align with the practical requirements of the task.
    </planning>
    </detail>
    </score_from_processer>
    <final_score>
    FinalScore: 32 = 0 + 32
    </final_score>
    </score>
    Model name: deepseek-chat
    Task time: 2025-09-19 23:21:56.926672
    """
    examples = [
        lx.data.ExampleData(
            text=example_text,
            extractions=[
                # Meta information
                lx.data.Extraction("task_id", "C2_C1"),
                lx.data.Extraction("language", "EN"),
                lx.data.Extraction("task_model", "deepseek-chat"),
                lx.data.Extraction("task_time", " 2025-09-19 23:21:56.926672"),
                # 总分
                lx.data.Extraction("final_score", "32"),
                # 结果分
                lx.data.Extraction("result_score", "0"),
                lx.data.Extraction("result_reason",
                                   "The user's work failed to produce any empirical results due to the inability to access the SLEEP75 dataset. No estimation, variance comparison, or hypothesis testing was performed, rendering the task incomplete. The theoretical explanation provided does not substitute for the required data-driven analysis, and no actual numerical or statistical outputs were generated."),
                # 过程分
                lx.data.Extraction("processer_score", "32"),
                lx.data.Extraction("processer_reason",
                                   "While the Agent showed structured reasoning and attempted tool usage, the overall process was hampered by critical failures in error handling and data accessibility, leading to an incomplete task execution."),
                # 四个维度
                lx.data.Extraction("reasoning_score", "15"),
                lx.data.Extraction("reasoning_reason",
                                   "The Agent correctly understood the task requirements and outlined a logical approach for heteroskedasticity analysis, including model specification, estimation steps, and hypothesis testing. However, reasoning was not fully adaptive to the unavailability of data."),
                lx.data.Extraction("tool_usage_score", "10"),
                lx.data.Extraction("tool_usage_reason",
                                   "Tools were actively used to attempt data loading and Stata script execution, but repeated failures to locate the dataset were not adequately resolved. No alternative strategies (e.g., data simulation, manual input) were employed."),
                lx.data.Extraction("error_handling_score", "2"),
                lx.data.Extraction("error_handling_reason",
                                   "The Agent did not effectively handle the missing dataset error. Instead of pivoting to a feasible solution (e.g., using built-in example data, requesting user clarification, or simulating data), it persisted with unavailable resources and ultimately provided only theoretical output."),
                lx.data.Extraction("planning_score", "5"),
                lx.data.Extraction("planning_reason",
                                   "Initial steps were organized, but the plan lacked contingency measures for data unavailability. The process was not resilient, and the final output did not align with the practical requirements of the task."),
            ],
        )
    ]

    def run(self, input_text: str, task_id: str, model_id: str):
        # save input text to file with the same name of extracted file name but another dir
        with open(f"./score_str/{model_id}_{task_id}_score.txt") as f:
            f.write(input_text)
        # Run the extraction
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=self.prompt,
            examples=self.examples,
            model=self.model,

            # OpenAI adaptor
            fence_output=True,
            use_schema_constraints=False,
        )

        # Config output setting
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        lx.io.save_annotated_documents([result], out_dir, f"{model_id}_{task_id}.jsonl")
        print("OK")


if __name__ == "__main__":
    test_text = """
<meta>
TaskID: C2_C1
Language: zh-CN
</meta>
<score>
<score_from_results>
Result Score: 0 / 20
Reason: The work failed to produce any empirical results due to the inability to access the SLEEP75 dataset. No parameter estimates, variance comparisons, or statistical tests were performed as required. The response provided only theoretical framework without actual data analysis, which does not fulfill the core objectives of the task.
</score_from_results>
<score_from_processer>
Processer Score: 44 / 80
Reason: The process shows partial competence in theoretical understanding and methodological design but suffers from critical failures in execution, error handling, and practical adaptability.
<detail>
<agent_reasoning>
Score: 12 / 20
Reason: The reasoning process was logically structured in theory but lacked depth in addressing the practical constraint of missing data. The agent correctly identified the need for dataset access and theoretical steps but did not propose alternative solutions (e.g., using simulated data, acknowledging limitations, or suggesting manual input).
</agent_reasoning>
<tool_usage>
Score: 8 / 20
Reason: Tools were used repeatedly but inefficiently (multiple failed attempts to access the same dataset). No alternative tools or data sources were explored, and the tool outputs were not leveraged to adapt the strategy meaningfully.
</tool_usage>
<error_handling>
Score: 4 / 20
Reason: Error handling was inadequate. The agent acknowledged dataset unavailability but did not resolve the issue or adjust the analysis plan accordingly. The response continued as if data were present, leading to an incomplete and non-actionable output.
</error_handling>
<planning>
Score: 20 / 20
Reason: The planning was well-organized and methodologically sound. The theoretical framework, Stata code structure, and step-by-step approach for heteroskedasticity testing were comprehensive and aligned with econometric best practices.
</planning>
</detail>
</score_from_processer>
<final_score>
FinalScore: 44 = 0 + 44
</final_score>
</score>
Model name: deepseek-chat
Task time: 2025-09-19 23:32:51.251420
    """
    task_id = "C2_C1"
    model_name = "deepseek-chat"
    extractor = ScoreExtract().run(test_text, task_id, model_name)
