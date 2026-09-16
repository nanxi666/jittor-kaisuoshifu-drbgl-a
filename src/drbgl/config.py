"""配置读写、随机种子、日志与运行命令工具（可复现性基础设施）。

对外接口::

    load_config(path)            读取 YAML 配置
    deep_update(base, override)  把命令行覆盖项合并进配置（命令行优先级更高）
    save_config(config, path)    把本次运行实际使用的配置落盘
    save_command(command, path)  把本次运行的命令落盘
    set_seed(seed)               统一设置 Python / NumPy / Jittor 随机种子
    tee_stdout(path)             上下文管理器：把 stdout 同时写入日志文件
    setup_logger(name, path)     返回同时输出到终端与日志文件的 logger
    require_files(paths, hint)   缺失文件时抛出带修复提示的 FileNotFoundError
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。

    Args:
        path: 配置文件路径。
    Returns:
        配置字典。
    Raises:
        FileNotFoundError: 配置文件不存在时给出路径提示。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return config


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """把 ``override`` 中非 None 的值覆盖到 ``base``（支持嵌套字典，原地修改）。

    Args:
        base: 被覆盖的配置字典。
        override: 覆盖项，值为 None 时忽略，便于直接传命令行参数。
    Returns:
        合并后的 ``base``。
    """
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def to_plain(value: Any) -> Any:
    """把配置值转成 YAML 可序列化的形态（Path -> str，递归处理容器）。

    Args:
        value: 任意配置值。
    Returns:
        只含 dict / list / 标量的等价结构。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """把本次运行实际使用的配置落盘为 YAML。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            to_plain(config), handle, allow_unicode=True, sort_keys=False
        )


def save_command(path: str | Path, command: Sequence[str] | None = None) -> None:
    """把本次运行的命令行写入 command.txt。

    Args:
        path: 目标文件路径，通常是 `<output_dir>/command.txt`。
        command: 命令序列，默认取 `sys.argv`。
    """
    parts = list(command) if command is not None else sys.argv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    """统一设置随机种子。

    Args:
        seed: 随机种子，同时作用于 Python random、NumPy 与 Jittor。

    Jittor 不可用时（例如纯 NumPy 推理路径、或 Jittor 尚未完成一次性 JIT 编译）
    给出告警但不中断，因为该路径不使用 Jittor 随机数。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import jittor as jt
    except Exception as error:  # Jittor 缺失 / 未完成首次编译 / 工具链不可用
        warnings.warn(
            f"jittor seed not applied ({type(error).__name__}: {error}); "
            "only Python/NumPy seeds were set; fix jittor before retraining",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(seed)
    else:  # pragma: no cover - 兼容不同 Jittor 版本
        jt.misc.set_global_seed(seed)


class _Tee:
    """把写入转发到多个流，用于同时输出到终端与日志文件。"""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def tee_stdout(path: str | Path) -> Iterator[Path]:
    """在上下文内把 stdout / stderr（含第三方 print 与异常回溯）写入日志文件。

    Args:
        path: 日志文件路径。
    Yields:
        日志文件路径。
    """
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handle = stack.enter_context(path.open("w", encoding="utf-8"))
        stack.enter_context(redirect_stdout(_Tee(sys.stdout, handle)))
        stack.enter_context(redirect_stderr(_Tee(sys.stderr, handle)))
        yield path


def setup_logger(name: str, path: str | Path) -> Any:
    """返回同时写终端与日志文件的 logger。

    Args:
        name: logger 名称。
        path: 日志文件路径。
    Returns:
        配置好的 `logging.Logger`。
    """
    import logging

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def require_files(
    paths: Sequence[Path],
    hint: str = "",
) -> list[Path]:
    """校验一组文件存在，缺失时给出可修复的报错。

    Args:
        paths: 待校验的路径。
        hint: 出错时附加的修复建议。
    Returns:
        原样返回输入路径列表，便于链式使用。
    Raises:
        FileNotFoundError: 任一文件缺失时抛出，消息中列出全部缺失路径。
    """
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        message = "missing required files:\n  " + "\n  ".join(missing)
        if hint:
            message += f"\nhow to fix: {hint}"
        raise FileNotFoundError(message)
    return list(paths)
