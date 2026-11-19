import pytest
from agents.base_agent import Agent, AgentMaxStepsError, ToolCall, LLMResponse


class MockLLMClient:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_messages: list[list[dict]] = []

    def generate(self, messages, system_instruction, tools=None) -> LLMResponse:
        self.call_count += 1
        self.recorded_messages.append([dict(m) for m in messages])
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(text="Fallback response")


def test_normal_final_answer_no_tools():
    mock_client = MockLLMClient([
        LLMResponse(text="Analysis complete. All requirements parsed.")
    ])
    agent = Agent(
        name="TestAgent",
        system_prompt="You are a test agent.",
        tools=[],
        client=mock_client
    )

    result = agent.run("Please analyze this text.")
    assert result == "Analysis complete. All requirements parsed."
    assert mock_client.call_count == 1


def test_one_tool_call_then_final_answer():
    def get_user_skills() -> list[str]:
        return ["Python", "FastAPI"]

    mock_client = MockLLMClient([
        # Step 1: Model requests tool call
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="get_user_skills", args={})]
        ),
        # Step 2: Model returns final answer using tool result
        LLMResponse(
            text="Found skills: Python, FastAPI.",
            tool_calls=[]
        )
    ])

    agent = Agent(
        name="ToolUserAgent",
        system_prompt="Use the tool.",
        tools=[get_user_skills],
        client=mock_client
    )

    result = agent.run("What skills do I have?")
    assert result == "Found skills: Python, FastAPI."
    assert mock_client.call_count == 2

    # Verify tool response was fed back in messages
    last_turn = mock_client.recorded_messages[-1]
    assert any(m.get("role") == "tool" and m.get("content") == {"result": ["Python", "FastAPI"]} for m in last_turn)


def test_unknown_tool_recovers():
    mock_client = MockLLMClient([
        # Step 1: Model requests unknown tool
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="non_existent_tool", args={"foo": "bar"})]
        ),
        # Step 2: Model inspects error and returns final answer
        LLMResponse(
            text="Recovered from unknown tool.",
            tool_calls=[]
        )
    ])

    agent = Agent(
        name="RecoveryAgent",
        system_prompt="System instructions.",
        tools=[],
        client=mock_client
    )

    result = agent.run("Execute something.")
    assert result == "Recovered from unknown tool."
    assert mock_client.call_count == 2

    # Verify error message was passed back
    second_call_messages = mock_client.recorded_messages[1]
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "error" in tool_msg["content"]
    assert "Unknown tool 'non_existent_tool'" in tool_msg["content"]["error"]


def test_invalid_arguments_recovers_via_pydantic():
    def multiply_number(value: int) -> int:
        return value * 2

    mock_client = MockLLMClient([
        # Step 1: Model sends string instead of integer
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="multiply_number", args={"value": "invalid_int"})]
        ),
        # Step 2: Model recovers from validation error
        LLMResponse(
            text="Corrected and handled invalid arguments.",
            tool_calls=[]
        )
    ])

    agent = Agent(
        name="ValidatorAgent",
        system_prompt="System instructions.",
        tools=[multiply_number],
        client=mock_client
    )

    result = agent.run("Multiply.")
    assert result == "Corrected and handled invalid arguments."
    assert mock_client.call_count == 2

    # Verify Pydantic validation error was captured
    second_call_messages = mock_client.recorded_messages[1]
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "error" in tool_msg["content"]
    assert "Invalid arguments for tool 'multiply_number'" in tool_msg["content"]["error"]


def test_tool_raising_exception_recovers():
    def faulty_tool() -> None:
        raise RuntimeError("Database connection timed out")

    mock_client = MockLLMClient([
        # Step 1: Calls faulty tool
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="faulty_tool", args={})]
        ),
        # Step 2: Recovers
        LLMResponse(
            text="Gracefully handled tool failure.",
            tool_calls=[]
        )
    ])

    agent = Agent(
        name="FaultTolerantAgent",
        system_prompt="System instructions.",
        tools=[faulty_tool],
        client=mock_client
    )

    result = agent.run("Run faulty operation.")
    assert result == "Gracefully handled tool failure."
    assert mock_client.call_count == 2

    second_call_messages = mock_client.recorded_messages[1]
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "error" in tool_msg["content"]
    assert "Database connection timed out" in tool_msg["content"]["error"]


def test_max_steps_exceeded_raises_error():
    def endless_tool() -> str:
        return "looping"

    # Client constantly asks to execute tool, never giving a final answer
    infinite_responses = [
        LLMResponse(text=None, tool_calls=[ToolCall(name="endless_tool", args={})])
        for _ in range(10)
    ]
    mock_client = MockLLMClient(infinite_responses)

    agent = Agent(
        name="LooperAgent",
        system_prompt="System instructions.",
        tools=[endless_tool],
        client=mock_client,
        max_steps=5
    )

    with pytest.raises(AgentMaxStepsError) as exc_info:
        agent.run("Start infinite loop.")

    assert "exceeded maximum allowed steps (5)" in str(exc_info.value)
    assert mock_client.call_count == 5
