#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : utils.py

from typing import Dict, List, Any

from langchain_core.messages import BaseMessage


def resort_ai_msg(messages: List[BaseMessage]) -> Dict[str, str]:
    """
    Make the AI_Message change to AI-Agent's processer and result
    """

    def _stringify(content: Any) -> str:
        # LangChain content 可能是 str / list[dict|str] / None
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for it in content:
                if isinstance(it, str):
                    parts.append(it)
                elif isinstance(it, dict):
                    # 尽量抽取常见字段
                    for k in ("text", "content", "output", "result", "message"):
                        v = it.get(k)
                        if isinstance(v, str):
                            parts.append(v)
                            break
                    else:
                        parts.append(str(it))
                else:
                    parts.append(str(it))
            return "\n".join(p for p in parts if p)
        # 兜底
        return str(content)

    # 找到最后一个 AI 消息，作为 result
    last_ai_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "ai":
            last_ai_idx = i
            break

    # 过程部分：不包含最后一个 AI 消息
    process_slice = messages[:last_ai_idx] if last_ai_idx is not None else messages

    lines: List[str] = []
    for m in process_slice:
        mtype = getattr(m, "type", "")
        if mtype == "system":
            continue  # 忽略 system
        if mtype == "human":
            header = "# Human MSG"
        elif mtype == "ai":
            header = "# AI MSG"
        elif mtype in ("tool", "function"):
            header = "# Tool MSG"
        else:
            # 其他类型忽略，避免破坏格式
            continue

        body = _stringify(getattr(m, "content", ""))
        # 保证都是 str
        body = "" if body is None else str(body)
        lines.append(f"{header}\n{body}".rstrip())

    processer_str = "\n".join(lines).strip()

    # 结果部分：最后一个 AI 的文本
    result_str = ""
    if last_ai_idx is not None:
        result_str = _stringify(getattr(messages[last_ai_idx], "content", "")).strip()

    return {
        "processer": processer_str,
        "result": result_str,
    }
