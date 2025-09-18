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

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from exp_client import system_prompt


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

    async def run(self, task: str):
        mcp_tools = await self.mcp_client.get_tools()
        agent = create_react_agent(
            model=self.llm,
            tools=mcp_tools,
            prompt=system_prompt
        )
        result = await agent.ainvoke({"messages": task})
        return result
