import json
import logging
import os
import re
from typing import Any, Callable, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, validate_call

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentMaxStepsError(Exception):
    pass


class AgentValidationError(Exception):
    pass


def extract_json_payload(text: str) -> dict[str, Any] | list[Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    return json.loads(cleaned)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class GeminiClientWrapper:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._client = None

    def __repr__(self) -> str:
        return f"<GeminiClientWrapper model='{self.model}'>"


    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set. Set it in .env or pass it explicitly.")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_instruction: str,
        tools: list[Callable] | None = None
    ) -> LLMResponse:
        from google.genai import types

        client = self._get_client()
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role")
            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.get("content", ""))]
                    )
                )
            elif role == "model":
                parts: list[types.Part] = []
                if msg.get("text"):
                    parts.append(types.Part.from_text(text=msg["text"]))
                for tc in msg.get("tool_calls", []):
                    name = tc.name if isinstance(tc, ToolCall) else tc["name"]
                    args = tc.args if isinstance(tc, ToolCall) else tc.get("args", {})
                    parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg["name"],
                                response=msg["content"]
                            )
                        ]
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools or [],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.0
        )


        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        tool_calls: list[ToolCall] = []
        if response.function_calls:
            for fc in response.function_calls:
                tool_calls.append(ToolCall(name=fc.name, args=fc.args or {}))

        return LLMResponse(text=response.text, tool_calls=tool_calls)


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Callable] | None = None,
        client: Any | None = None,
        max_steps: int = 5
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tools_map: dict[str, Callable] = {tool.__name__: tool for tool in self.tools}
        self.client = client or GeminiClientWrapper()
        self.max_steps = max_steps

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.tools_map:
            available = list(self.tools_map.keys())
            return {"error": f"Unknown tool '{name}'. Available tools: {available}"}

        target_tool = self.tools_map[name]
        try:
            # Validate arguments against tool function signature using Pydantic
            validated_tool = validate_call(target_tool)
            result = validated_tool(**args)
            return {"result": result}
        except ValidationError as e:
            return {"error": f"Invalid arguments for tool '{name}': {str(e)}"}
        except Exception as e:
            return {"error": f"Tool '{name}' failed with error: {str(e)}"}

    def run(self, user_prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        logger.info("Agent [%s] started run with user prompt length: %d chars", self.name, len(user_prompt))

        for step in range(1, self.max_steps + 1):
            logger.info("Agent [%s] Step %d/%d: Calling model", self.name, step, self.max_steps)
            response: LLMResponse = self.client.generate(
                messages=messages,
                system_instruction=self.system_prompt,
                tools=self.tools
            )

            # Check if model requested tools
            if response.tool_calls:
                messages.append({
                    "role": "model",
                    "text": response.text,
                    "tool_calls": response.tool_calls
                })

                for tc in response.tool_calls:
                    logger.info("Agent [%s] Step %d: Requesting tool '%s' with args %s",
                                self.name, step, tc.name, tc.args)

                    tool_output = self._execute_tool(tc.name, tc.args)
                    summary = str(tool_output)[:120]
                    logger.info("Agent [%s] Step %d: Tool '%s' executed. Result summary: %s",
                                self.name, step, tc.name, summary)

                    messages.append({
                        "role": "tool",
                        "name": tc.name,
                        "content": tool_output
                    })
            else:
                # Model returned a final textual response
                final_text = response.text or ""
                logger.info("Agent [%s] Step %d: Finished with final response (%d chars)",
                            self.name, step, len(final_text))
                return final_text

        logger.error("Agent [%s] reached maximum steps (%d) without completing", self.name, self.max_steps)
        raise AgentMaxStepsError(
            f"Agent '{self.name}' exceeded maximum allowed steps ({self.max_steps}) without returning a final answer."
        )

    def run_structured(self, user_prompt: str, response_model: type[T]) -> T:
        raw_output = self.run(user_prompt)
        try:
            payload = extract_json_payload(raw_output)
            return response_model.model_validate(payload)
        except Exception as first_err:
            logger.warning(
                "Agent [%s] output failed validation: %s. Retrying once.",
                self.name,
                first_err
            )
            retry_prompt = (
                f"Your previous response failed validation with error: {first_err}.\n"
                f"Required schema: {json.dumps(response_model.model_json_schema())}\n"
                "Please return ONLY a valid JSON object matching the required schema."
            )
            retry_output = self.run(retry_prompt)
            try:
                retry_payload = extract_json_payload(retry_output)
                return response_model.model_validate(retry_payload)
            except Exception as second_err:
                logger.error("Agent [%s] failed validation after retry: %s", self.name, second_err)
                raise AgentValidationError(
                    f"Agent '{self.name}' failed to produce valid {response_model.__name__} after retry: {second_err}"
                ) from second_err

