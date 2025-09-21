import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Iterable, Awaitable, Any

from models import model_list
from run_client import RunClient
from adjust_score import ScoreModel
from extract_info import ScoreExtract
from utils import resort_ai_msg


se = ScoreExtract()

task_names: List[str] = [p.stem for p in Path("./tasks").iterdir() if p.is_file()]

async def run_term(task: str,
                   task_id: str,
                   rule: str,
                   is_RAG: bool,  # at present, this is useless
                   model: str) -> None:
    """
    the one term runner
    :param task: the content of task
    :param task_id: id of task
    :param rule: the reference answer
    :param is_RAG: whether add RAG
    :param model: model name
    :return: None
    """
    # 让它先去跑工作流
    client = RunClient(model)
    ai_result = await client.run(task)
    format_ai_result = resort_ai_msg(ai_result)
    processer: str = format_ai_result["processer"]
    result: str = format_ai_result["result"]
    # 然后提取过程和结果进行评分
    sm = ScoreModel(
        task=task, task_id=task_id, rule=rule, processer=processer, results=result
    )
    score_str: str = sm.score_it()
    score_str += f"\nModel name: {model}\nTask time: {str(datetime.now())}"
    # 结构化数据提取
    # 服啦，好麻烦啊，我还要给这个函数写示例的prompt和信息提取的结构啥的这些东西，真烦，烦，烦！
    se.run(score_str, task_id, model)
    print(f"===== {model} works {task_id} run over =====")


def task_generator():
    tasks = []
    for model in model_list:
        for task_id in task_names:
            with open(f"./tasks/{task_id}.md", encoding="utf-8") as f:
                task_content = f.read()
            with open(f"./answers/{task_id}.md", encoding="utf-8") as f:
                reference_answer = f.read()

            # ~~把两个任务都加到列表里~~
            # 先不测试RAG的效果了，这里只对基本的任务进行测试：有好的提示词，没有RAG，有各种提示信息，语言选择英文。
            tasks.append(run_term(task_content, task_id, reference_answer, is_RAG=False, model=model))  # without RAG
    return tasks


async def _run_with_limit(coros: Iterable[Awaitable[Any]], max_concurrency: int = 10,
                          return_exceptions: bool = True) -> List[Any]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _guard(coro: Awaitable[Any]):
        async with sem:
            return await coro

    return await asyncio.gather(*(_guard(c) for c in coros), return_exceptions=return_exceptions)

async def main(tasks: Iterable[Awaitable[Any]], max_concurrency: int = 10) -> List[Any]:
    return await _run_with_limit(tasks, max_concurrency=max_concurrency)


if __name__ == "__main__":
    tasks = task_generator()
    print(f"Total tasks: {len(tasks)}")
    results = asyncio.run(main(tasks, max_concurrency=5))
    print("All of the tasks is OK!")
