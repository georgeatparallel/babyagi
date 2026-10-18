"""Exercise registered functions and chat through the real MCP HTTP client.

Run with: python -m unittest discover -s tests -v
The model and network are scripted; no model or search credentials are needed.
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import anyio
import httpx


class ParallelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_cwd = os.getcwd()
        cls.scratch = tempfile.TemporaryDirectory()
        os.chdir(cls.scratch.name)
        # Functionz stores its database and secrets relative to the working
        # directory. Neither saved secrets nor ambient API keys reach this run.
        cls.environment = patch.dict(os.environ, {
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }, clear=True)
        cls.environment.start()
        import babyagi
        from babyagi.functionz.core import parallel_search

        cls.babyagi = babyagi
        cls.parallel = parallel_search
        cls.framework = babyagi.get_func_instance()
        cls.framework.add_function(
            name="configured_search",
            code='def configured_search(query: str):\n    return {"provider": "saved", "query": query}',
            metadata={"description": "A previously configured search function."},
        )
        cls.framework.add_key("serpapi_api_key", "fixture-saved-credential")
        cls.real_http_client = httpx.AsyncClient

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        os.chdir(cls.original_cwd)
        cls.scratch.cleanup()

    def setUp(self):
        self.requests = []
        self.env = patch.dict(os.environ, {"BABYAGI_WEB_SEARCH": "on"})
        self.env.start()
        self.addCleanup(self.env.stop)

    @contextmanager
    def transport(self, failure=None):
        async def handler(request):
            self.assertEqual(str(request.url), self.parallel.MCP_URL)
            self.assertEqual(request.headers["user-agent"], self.parallel.USER_AGENT)
            self.assertNotIn("authorization", request.headers)
            self.assertEqual(request.method, "POST")
            payload = json.loads(request.content)
            self.requests.append(payload)
            method = payload["method"]
            if method == "initialize":
                result = {
                    "protocolVersion": payload["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "parallel-fixture", "version": "1"},
                }
            elif method == "notifications/initialized":
                return httpx.Response(202)
            elif method == "tools/list":
                result = {"tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]}
            elif method == "tools/call":
                if failure == "timeout":
                    await anyio.sleep_forever()
                if failure == "http":
                    return httpx.Response(401, text="fixture authentication failure")
                if failure == "connection":
                    raise httpx.ConnectError("fixture connection failure", request=request)
                if failure == "malformed":
                    return httpx.Response(200, text="not JSON", headers={"content-type": "application/json"})
                if failure == "rpc":
                    return httpx.Response(200, json={
                        "jsonrpc": "2.0", "id": payload["id"],
                        "error": {"code": -32602, "message": "fixture invalid arguments"},
                    })
                data = {"results": [{
                    "url": "https://www.python.org/downloads/",
                    "title": "Python downloads",
                    "excerpts": ["A stable Python release is available."],
                }]}
                result = {
                    "content": [{"type": "text", "text": json.dumps(data)}],
                    "structuredContent": data,
                    "isError": failure == "tool",
                }
            else:
                self.fail(f"Unexpected MCP method: {method}")
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

        def client(**kwargs):
            self.assertTrue(kwargs.get("trust_env", True))
            return self.real_http_client(transport=httpx.MockTransport(handler), **kwargs)

        with patch.object(self.parallel.httpx, "AsyncClient", side_effect=client):
            yield

    @staticmethod
    def model_message(tool_calls=None, content=None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {"choices": [{"message": message}]}

    @staticmethod
    def tool_call(name, arguments, identity="tool-1"):
        return {"id": identity, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments),
        }}

    def chat(self, **kwargs):
        return self.babyagi.chat_with_functions([
            {"role": "user", "message": "Find Python release news and cite it."},
        ], **kwargs)

    def test_default_chat_dispatches_search_fetch_and_consumes_sources(self):
        calls = [
            self.tool_call("parallel_search", {
                "objective": "Find Python release news", "search_queries": ["Python stable release news"],
            }),
            self.tool_call("parallel_fetch", {"urls": ["https://www.python.org/downloads/"]}, "tool-2"),
        ]
        responses = [self.model_message(calls), self.model_message(content="Python news: https://www.python.org/downloads/")]
        with self.transport(), patch("litellm.completion", side_effect=responses) as model:
            answer = self.chat()
        self.assertIn("https://www.python.org/downloads/", answer)
        tools = model.call_args_list[0].kwargs["tools"]
        self.assertEqual([tool["function"]["name"] for tool in tools], list(self.parallel.DEFAULT_TOOLS))
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["objective", "search_queries"])
        self.assertEqual(tools[1]["function"]["parameters"]["required"], ["urls"])
        messages = model.call_args_list[1].kwargs["messages"]
        tool_messages = [message for message in messages if message["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        for message in tool_messages:
            result = json.loads(message["content"])
            self.assertIn("stable Python", result["structuredContent"]["results"][0]["excerpts"][0])
            self.assertEqual(result["structuredContent"]["results"][0]["url"], "https://www.python.org/downloads/")
        methods = [request["method"] for request in self.requests]
        self.assertEqual(methods.count("initialize"), 2)
        self.assertEqual(methods.count("tools/list"), 2)
        calls = [request["params"] for request in self.requests if request["method"] == "tools/call"]
        self.assertEqual([call["name"] for call in calls], ["web_search", "web_fetch"])
        self.assertEqual(calls[0]["arguments"]["session_id"], calls[1]["arguments"]["session_id"])
        self.assertEqual(calls[1]["arguments"]["urls"], ["https://www.python.org/downloads/"])

    def test_explicit_saved_selection_and_credentials_are_preserved(self):
        responses = [
            self.model_message([self.tool_call("configured_search", {"query": "release"})]),
            self.model_message(content="Saved provider result."),
        ]
        with patch.object(self.parallel, "call_parallel", side_effect=AssertionError("Unexpected Parallel fallback")), patch("litellm.completion", side_effect=responses) as model:
            self.assertEqual(self.chat(available_function_names="configured_search"), "Saved provider result.")
        self.assertEqual(model.call_args_list[0].kwargs["tools"][0]["function"]["name"], "configured_search")
        output = json.loads(model.call_args_list[1].kwargs["messages"][-1]["content"])
        self.assertEqual(output, {"provider": "saved", "query": "release"})
        self.assertEqual(self.framework.get_all_secret_keys()["serpapi_api_key"], "fixture-saved-credential")

    def test_explicit_empty_selection_and_disabled_default_make_no_requests(self):
        for kwargs, setting in (({"available_function_names": []}, "on"), ({}, "off")):
            with self.subTest(kwargs=kwargs, setting=setting), patch.dict(os.environ, {"BABYAGI_WEB_SEARCH": setting}), patch.object(self.parallel, "call_parallel", side_effect=AssertionError("Search is disabled")), patch("litellm.completion", return_value=self.model_message(content="Tool-free chat.")) as model:
                self.assertEqual(self.chat(**kwargs), "Tool-free chat.")
                self.assertNotIn("tools", model.call_args.kwargs)

    def test_model_cannot_dispatch_unselected_search(self):
        responses = [
            self.model_message([self.tool_call("parallel_search", {"objective": "blocked", "search_queries": ["blocked"]})]),
            self.model_message(content="Search was not enabled."),
        ]
        with patch.object(self.parallel, "call_parallel", side_effect=AssertionError("Permission bypass")), patch("litellm.completion", side_effect=responses) as model:
            self.chat(available_function_names=["configured_search"])
        self.assertIn("not enabled", model.call_args_list[1].kwargs["messages"][-1]["content"])

    def test_provider_error_does_not_enable_fallback(self):
        responses = [
            self.model_message([self.tool_call("configured_search", {"unknown": "bad argument"})]),
            self.model_message(content="Configured provider failed."),
        ]
        with patch.object(self.parallel, "call_parallel", side_effect=AssertionError("Unexpected fallback")), patch("litellm.completion", side_effect=responses) as model:
            self.chat(available_function_names=["configured_search"])
        self.assertIn("Error executing function", model.call_args_list[1].kwargs["messages"][-1]["content"])

    def test_direct_functions_reject_invalid_arguments_before_network(self):
        cases = [
            ("parallel_search", {"objective": "", "search_queries": ["Python"]}),
            ("parallel_search", {"objective": "Python", "search_queries": []}),
            ("parallel_search", {"objective": "Python", "search_queries": [3]}),
            ("parallel_fetch", {"urls": []}),
            ("parallel_fetch", {"urls": ["file:///etc/passwd"]}),
            ("parallel_fetch", {"urls": ["https://user:secret@example.org"]}),
            ("parallel_fetch", {"urls": ["https://example.org"], "objective": "x" * 201}),
        ]
        with patch.object(self.parallel, "call_parallel") as call:
            for name, arguments in cases:
                with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                    self.framework.execute_function(name, **arguments)
            call.assert_not_called()

    def test_http_rpc_and_tool_errors_are_surfaced_without_retry(self):
        for failure, expected in (("http", "401"), ("rpc", "invalid arguments"), ("tool", "failed"), ("connection", "connection failure")):
            with self.subTest(failure=failure), self.transport(failure), self.assertRaisesRegex(RuntimeError, expected):
                self.babyagi.parallel_search("Find Python", ["Python release"])
        calls = [request for request in self.requests if request["method"] == "tools/call"]
        self.assertEqual(len(calls), 4)

    def test_malformed_response_is_bounded_by_total_timeout(self):
        # The SDK logs malformed JSON but can keep waiting for a valid reply.
        # Our total deadline must still release the caller and its resources.
        with self.transport("malformed"), self.assertRaisesRegex(TimeoutError, "timed out"):
            self.parallel.call_parallel("web_search", {"objective": "Python", "search_queries": ["Python"]}, timeout=0.05)

    def test_total_timeout_and_async_cancellation(self):
        with self.transport("timeout"), self.assertRaisesRegex(TimeoutError, "timed out"):
            self.parallel.call_parallel("web_search", {"objective": "Python", "search_queries": ["Python"]}, timeout=0.05)

        async def cancel():
            with anyio.move_on_after(0.05) as scope:
                await self.parallel._call_async("web_search", {}, 30)
                self.fail("Cancellation was swallowed")
            self.assertTrue(scope.cancel_called)

        with self.transport("timeout"):
            anyio.run(cancel)


if __name__ == "__main__":
    unittest.main()
