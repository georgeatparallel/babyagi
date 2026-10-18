"""Anonymous Parallel Search MCP access for registered BabyAGI functions."""

import uuid
from datetime import timedelta

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "BabyAGI (+https://github.com/yoheinakajima/babyagi)"
# Keep one identifier across searches and fetches in this runtime. Never rotate
# identifiers or retry requests to work around the anonymous service's limits.
SESSION_ID = str(uuid.uuid4())
DEFAULT_TOOLS = ("parallel_search", "parallel_fetch")


def default_search_tools():
    """Only omitted chat selections use these defaults; explicit lists win."""
    import os

    if os.getenv("BABYAGI_WEB_SEARCH", "on").lower() in ("0", "false", "off", "no"):
        return []
    return list(DEFAULT_TOOLS)


def _error_text(error):
    # SDK transports can wrap HTTP failures in task-group exception groups.
    children = getattr(error, "exceptions", None)
    if children:
        return "; ".join(_error_text(child) for child in children)
    return str(error)


async def _call_async(tool_name, arguments, timeout):
    try:
        with anyio.fail_after(timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=httpx.Timeout(timeout),
            ) as client:
                async with streamable_http_client(MCP_URL, http_client=client) as streams:
                    async with ClientSession(
                        streams[0], streams[1],
                        read_timeout_seconds=timedelta(seconds=timeout),
                    ) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        if tool_name not in {tool.name for tool in tools.tools}:
                            raise RuntimeError(f"Parallel MCP does not offer {tool_name}.")
                        result = await session.call_tool(tool_name, arguments)
                        if result.isError:
                            detail = "\n".join(
                                item.text for item in result.content if item.type == "text"
                            )
                            raise RuntimeError(f"Parallel MCP {tool_name} failed: {detail}")
                        # Preserve structured results, citations, excerpts and text
                        # blocks for both direct callers and the chat consumer.
                        return result.model_dump(mode="json", exclude_none=True)
    except TimeoutError as error:
        raise TimeoutError(f"Parallel MCP timed out after {timeout} seconds.") from error
    except Exception as error:
        raise RuntimeError(f"Parallel MCP request failed: {_error_text(error)}") from error


def call_parallel(tool_name, arguments, timeout=30):
    """Run the HTTP MCP client from the synchronous function executor.

    No search credentials are read from the environment or function database.
    HTTPX still honors the user's proxy and certificate settings.
    Cancellation and KeyboardInterrupt propagate; failures never select another
    provider or retry the request.
    """
    if tool_name not in ("web_search", "web_fetch"):
        raise ValueError("Unsupported Parallel MCP tool.")
    arguments = {**arguments, "session_id": SESSION_ID}
    return anyio.run(_call_async, tool_name, arguments, timeout)
