import os 
from anthropic import AnthropicVertex, Anthropic

project_id = "r2eg-441800"
# Where the model is running
region = "us-east5"

client = AnthropicVertex(project_id=project_id, region=region)
model_name = "claude-sonnet-4@20250514"
# client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# model_name = "claude-sonnet-4-20250514"


# Same tool definitions as before
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    },
    "cache_control": {"type": "ephemeral"}  # Add cache_control to tools
}

database_tool = {
    "name": "database_query",
    "description": "Query product database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute"
            }
        },
        "required": ["query"]
    },
    "cache_control": {"type": "ephemeral"}  # Add cache_control to tools
}



# First request with interleaved thinking enabled
response = client.messages.create(
    model=model_name,
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    tools=[calculator_tool, database_tool],
    # tool_choice="any",
    # Enable interleaved thinking with beta header
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
    messages=[{
        "role": "user",
        "content": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?",
        "cache_control": {"type": "ephemeral"}  # Add cache_control to user message
    }]
)

print("Initial response:")
thinking_blocks = []
tool_use_blocks = []

for block in response.content:
    if block.type == "thinking":
        thinking_blocks.append(block)
        print(f"Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Text: {block.text}")
    elif block.type == "tool_use":
        tool_use_blocks.append(block)
        print(f"Tool use: {block.name} with input {block.input}")

# First tool result (calculator)
calculator_result = "7500"  # 150 * 50

# Continue with first tool result
response2 = client.messages.create(
    model=model_name,
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    tools=[calculator_tool, database_tool],
    # tool_choice="any",
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
    messages=[
        {
            "role": "user",
            "content": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?",
            "cache_control": {"type": "ephemeral"}  # Add cache_control to user message
        },
        {
            "role": "assistant",
            "content": response.content,
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_blocks[0].id,
                "content": calculator_result
            }],
            "cache_control": {"type": "ephemeral"}  # Add cache_control to tool result message
        }
    ]
)

print("\nAfter calculator result:")
# With interleaved thinking, Claude can think about the calculator result
# before deciding to query the database
for block in response2.content:
    if block.type == "thinking":
        thinking_blocks.append(block)
        print(f"Interleaved thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Text: {block.text}")
    elif block.type == "tool_use":
        tool_use_blocks.append(block)
        print(f"Tool use: {block.name} with input {block.input}")

# Second tool result (database)
database_result = "5200"  # Example average monthly revenue

# Continue with second tool result
response3 = client.messages.create(
    model=model_name,
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    tools=[calculator_tool, database_tool],
    # tool_choice="any",
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
    messages=[
        {
            "role": "user",
            "content": "What's the total revenue if we sold 150 units of product A at $50 each, and how does this compare to our average monthly revenue from the database?",
            "cache_control": {"type": "ephemeral"}  # Add cache_control to user message
        },
        {
            "role": "assistant",
            "content": response.content
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_blocks[0].id,
                "content": calculator_result
            }],
            "cache_control": {"type": "ephemeral"}  # Add cache_control to first tool result
        },
        {
            "role": "assistant",
            "content": response2.content
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_blocks[1].id,
                "content": database_result
            }],
            "cache_control": {"type": "ephemeral"}  # Add cache_control to second tool result
        }
    ]
)

print("\nAfter database result:")
# With interleaved thinking, Claude can think about both results
# before formulating the final response
for block in response3.content:
    if block.type == "thinking":
        print(f"Final thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Final response: {block.text}")