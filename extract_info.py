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
    prompt = textwrap.dedent("""\
    提取报告中的“公司名称”和“关键风险点”。输出要基于原文，不要意译。
    """)


    # 2. Provide a high-quality example to guide the model
    examples = [
        lx.data.ExampleData(
            text="本公司（上海比翼视界）面临原材料价格波动风险。",
            extractions=[
                lx.data.Extraction("company", "上海比翼视界"),
                lx.data.Extraction("risk", "原材料价格波动")
            ],
        )
    ]

    def __init__(self):
        pass

    def run(self, input_text, task_id: str, model_id: str):
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
