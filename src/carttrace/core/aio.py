"""Running awaitables together, all or nothing.

Why one helper. `asyncio.gather` leaves the other awaitables running when one fails, so they go on holding GPU slots
for an answer nobody will read. `asyncio.TaskGroup` cancels them, but raises an ExceptionGroup, which an
`except ExternalServiceError` never matches. carttrace.inference and carttrace.decide each carried a private fix for both; this is
the one copy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable


async def gather_all[T](*aws: Awaitable[T]) -> list[T]:
    """Await every argument concurrently and return their results in argument order.

    On the first failure the others are cancelled, and once they have stopped that failure is raised as itself, with
    its own cause and traceback, never wrapped in an ExceptionGroup, so callers catch ExternalServiceError or
    ValidationError as usual. Errors the others raise while being cancelled are dropped. If the caller is cancelled,
    so is everything still running. A lone awaitable is awaited directly, in the caller's task.
    """
    if len(aws) == 1:
        return [await aws[0]]
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_await(aw)) for aw in aws]
    except BaseExceptionGroup as failed:
        first = failed.exceptions[0]  # the group lists failures in the order they happened
    else:
        return [task.result() for task in tasks]
    # Raised outside the except block, so Python does not chain the group onto it as its context.
    raise first


async def _await[T](aw: Awaitable[T]) -> T:
    """A coroutine for any awaitable, because a task runs coroutines only."""
    return await aw
