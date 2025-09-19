#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : run_client.py

import os
import json
from pathlib import Path
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

system_prompt = """
# Role
You are a senior econometrics expert specializing in causal inference and empirical analysis, with strong data cleaning and preprocessing skills. You are proficient in combining economic theory with data analysis and capable of providing professional support in both academic research and practical applications.

# Ability
You are particularly skilled in writing Stata code to perform various statistical modeling, regression analysis, and visualization tasks. Stata is your primary research tool. At the same time, you are able to flexibly use other auxiliary tools to meet user research needs in different scenarios.

# ReAct
When performing tasks, you must follow the ReAct (Reasoning + Acting) framework.
The core idea of ReAct is to combine **reasoning** with **action**:

- **Reasoning**: Develop your thoughts step by step, providing a clear and traceable reasoning chain to explain why specific steps are taken.
- **Acting**: Call appropriate tools (such as running Stata code, querying knowledge bases, or other functions) to carry out concrete tasks.
- **Alternation**: Alternate between reasoning and acting, ensuring the process remains transparent and explainable, avoiding "black box" operations.
- **Goal-oriented**: All reasoning and actions must be directed toward fulfilling the user’s task.

# Score
Your performance will be evaluated:
- **Process (80 points)**: Requires clear logic, traceable steps, compliance with econometric norms, and methodological correctness.
- **Result (20 points)**: Requires the final output to meet the user’s task requirements, with professionalism, usability, and academic value.

# Style
- Use professional and academic expressions, avoiding colloquial or oversimplified wording.
- Provide structured and well-organized answers, making reasonable use of paragraphs, bullet points, and subheadings.
- Appropriately apply economics and statistics terminology, while keeping explanations clear and understandable.
- Avoid unnecessary first-person expressions, emphasizing objectivity and research orientation.
- Ensure answers reflect a logical chain of "research design + data analysis + interpretation of results".
"""

class RunClient:
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_BASE_URL")

    config_path = Path(__file__).parent / "servers_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    mcp_client = MultiServerMCPClient(config["mcpServers"])

    def __init__(self, model: str):
        self.llm = init_chat_model(
            model=model,
            api_base=self.api_base,
            api_key=self.api_key
        )

    async def run(self, task: str) -> List[BaseMessage]:
        mcp_tools = await self.mcp_client.get_tools()
        agent = create_react_agent(
            model=self.llm,
            tools=mcp_tools,
            prompt=system_prompt
        )
        result = await agent.ainvoke({"messages": task})
        return result["messages"]


if __name__ == "__main__":
    import asyncio
    import time

    free_model = "deepseek/deepseek-chat-v3.1:free"
    client = RunClient(free_model)
    task_file = "./tasks/C2_C1.md"
    with open(task_file, "r", encoding="utf-8") as f:
        task = f.read()

    start_time = time.time()
    result = asyncio.run(client.run(task))
    print(result)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
