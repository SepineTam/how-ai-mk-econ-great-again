#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : api_test.py

import os

from langchain.chat_models import init_chat_model

# You can set your api_key and base_url here
# os.environ["OPENROUTER_API_KEY"] = "<YOUR_API_KEY>"
# os.environ["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"

llm = init_chat_model(
    model="deepseek/deepseek-chat-v3.1:free",  # this model is free to use
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base=os.getenv("OPENROUTER_BASE_URL"),
)

result = llm.invoke("Who are you? Answer me within 30 words")
print(result)
