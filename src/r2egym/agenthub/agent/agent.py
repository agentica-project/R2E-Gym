import os
import re
import copy
import yaml
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

import litellm
from openai import OpenAI
from anthropic import AnthropicVertex  # Add Anthropic Vertex import

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from r2egym.agenthub.tools import (
    search_tool,
    file_editor,
    bash_execute_tool,
    finish_tool,
    anthropic_file_editor,
    anthropic_bash_execute,
    anthropic_search,
    anthropic_finish,
)

logger = get_logger(__name__)  # Logger for this module

# Anthropic Vertex configuration
ANTHROPIC_PROJECT_ID = "r2eg-441800"  # Your GCP project ID
ANTHROPIC_REGION = "us-east5"  # Region where the model is running


##############################################################################
# AgentArgs Dataclass
##############################################################################
@dataclass
class AgentArgs:
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    llm_base_url: Optional[str] = "http://localhost:8000/v1"  # None
    demo_file: Optional[Path] = None
    other_args: Optional[Dict[str, Any]] = None  # To handle extra configurations

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


##############################################################################
# Agent Class
##############################################################################
class Agent:
    """Agent handles the behavior of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        # self.trajectory_steps: List[TrajectoryStep] = []
        if logger is None:
            self.logger = get_logger(name)  # initialize logger from the agent name
        else:
            self.logger = logger
        self.llm_name = args.llm_name
        self.llm_base_url = (
            # "http://localhost:8000/v1"
            "http://localhost:8000/v1"
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name)
            else None
        )
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.logger.info(f"Initialized Agent: {name} with LLM: {args.llm_name}")
        self.max_retries = self.other_args.get("max_retries", 3)
        self.llm_timeout = self.other_args.get("timeout", 120)

        self.anthropic_client = AnthropicVertex(
            project_id=ANTHROPIC_PROJECT_ID, region=ANTHROPIC_REGION
        )
        self.logger.info(f"Initialized Anthropic Vertex client for {self.llm_name}")

    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in placeholders."""
        system_prompt = self.system_prompt_template.format(
            # problem_statement=problem_statement,
            # structure=structure,
            command_docs=command_docs,
            demo=demo,
        )
        return system_prompt

    def prepare_instance_prompt(
        self, agent_history: str, command_docs: str, steps_remaining: int
    ) -> str:
        """Prepare the instance prompt by filling in placeholders."""
        instance_prompt = self.instance_prompt_template.format(
            agent_history=agent_history,
            command_docs=command_docs,
        )
        # self.logger.info(isinstance(steps_remaining, int))
        # Add steps remaining message
        if steps_remaining > 0:
            stepcount_message = f"Steps Remaining: {steps_remaining}"
        else:
            stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
        instance_prompt += f"\n{stepcount_message}"
        self.logger.info(stepcount_message)  # Log the steps remaining message
        return instance_prompt

    def prepare_history_message(self, include_all_obs=False) -> str:
        """Prepare the agent's message history as a string."""
        history = ""
        for idx, step in enumerate(self.trajectory_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            # history += f'THOUGHT:\n```\n{thought}\n```\n'
            # history += f'ACTION:\n```\n{action}\n```\n'
            action_template = """
            {thought}
            ```
            {action}
            ```
            """
            history += action_template.format(thought=thought, action=action)
            if idx == len(self.trajectory_steps) - 1 or include_all_obs:
                history += f"\nOBSERVATION:\n```\n{observation}\n```\n"
            # add a separator
            history += "-" * 50 + "\n"
        return history

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def condense_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Condense older user messages if total token usage exceeds a threshold.
        Replaces the content of those older user messages (after the first)
        with a placeholder until total tokens are under the limit.
        """
        MAX_TOKENS = 28000
        # Make a deepcopy so we don't mutate the original list
        messages_ = copy.deepcopy(messages)

        # Count the total tokens in the conversation so far
        total_tokens = self._count_tokens(messages_)
        self.logger.warning(
            f"Condensing history to save context. total tokens: {total_tokens}, max tokens: {MAX_TOKENS}"
        )
        if total_tokens <= MAX_TOKENS:
            return messages_

        # Identify user messages (role='user'), skipping the very first user message
        user_msg_indexes = [
            i for i, msg in enumerate(messages_) if msg["role"] == "user"
        ]
        if len(user_msg_indexes) <= 1:
            # If there's only 0 or 1 user messages total, nothing to condense
            return messages_

        # Start condensing from the second user message onward excluding last user message
        for idx in user_msg_indexes[1:-1]:
            # Replace with a short placeholder
            messages_[idx]["content"] = "<Observation condensed for saving context>"

            # Re-count tokens after condensing
            total_tokens = self._count_tokens(messages_)
            if total_tokens <= MAX_TOKENS:
                break

        self.logger.warning(
            f"Condensed history to save context. total tokens: {total_tokens}, max tokens: {MAX_TOKENS}"
        )

        return messages_

    def condense_history_old(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Condense older user messages if total token usage exceeds a threshold,
        but do so by targeting the longest user messages first.
        Skips the very first and very last user messages.
        """
        MAX_TOKENS = 28000
        messages_ = copy.deepcopy(messages)

        total_tokens = self._count_tokens(messages_)
        self.logger.warning(
            f"Condensing history to save context. total tokens: {total_tokens}, max tokens: {MAX_TOKENS}"
        )
        if total_tokens <= MAX_TOKENS:
            return messages_

        # Identify user messages (role='user')
        user_msg_indexes = [
            i for i, msg in enumerate(messages_) if msg["role"] == "user"
        ]

        # If there's only 0 or 1 user messages, or effectively no middle messages to condense, do nothing
        if len(user_msg_indexes) <= 1:
            return messages_

        # Indices of user messages to consider condensing (skip first and last user message)
        to_condense = user_msg_indexes[1:-1]
        if not to_condense:
            return messages_

        # Sort those middle user messages by descending length (longest first)
        to_condense_sorted = sorted(
            to_condense,
            key=lambda idx: self._count_tokens([messages_[idx]]),
            reverse=True,
        )

        # Condense from the longest to shortest until we are below the token limit
        for idx in to_condense_sorted:
            messages_[idx]["content"] = "<Observation condensed for saving context>"

            total_tokens = self._count_tokens(messages_)
            if total_tokens <= MAX_TOKENS:
                break

        self.logger.warning(
            f"Condensed history to save context. total tokens: {total_tokens}, max tokens: {MAX_TOKENS}"
        )

        return messages_

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count

    def model_query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        condense_history: bool = True,
    ) -> Dict[str, Any]:
        """Query the LLM with the messages and measure execution time."""
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            # Define tools in OpenAI format
            openai_tools = [search_tool, file_editor, bash_execute_tool, finish_tool]
            anthropic_tools = [
                anthropic_file_editor,
                anthropic_bash_execute,
                # anthropic_search,
                anthropic_finish,
            ]

            # # Add cache_control to tools for Vertex
            # for tool in anthropic_tools:
            #     tool["cache_control"] = {"type": "ephemeral"}

            # self.logger.warning(f"Using Anthropic Vertex with context caching for {self.llm_name}")

            # Add cache_control to messages
            # breakpoints_remaining = 3
            # for message in reversed(messages):
            #     if message["role"] in ("user", "tool"):
            #         if breakpoints_remaining > 0:
            #             message["cache_control"] = {"type": "ephemeral"}
            #             breakpoints_remaining -= 1
            #         else:
            #             break

        # Start timer
        start_time = time.time()
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        if not self.use_fn_calling and condense_history:
            # condense messages after first user message
            messages_ = self.condense_history(messages)
        else:
            messages_ = copy.deepcopy(messages)

        # query the model with retries
        while retries < self.max_retries:
            try:
                # Check if using Anthropic Vertex
                # if self.anthropic_client and "vertex" in self.llm_name.lower() and "claude" in self.llm_name.lower():
                self.logger.warning(
                    f"Using direct Anthropic Vertex client for {self.llm_name}"
                )
                model_name = (
                    self.llm_name.split("vertex-")[-1]
                    if "vertex-" in self.llm_name
                    else self.llm_name
                )

                # Extra headers for Anthropic features
                extra_headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}

                # Add thinking capability if needed
                thinking_config = {"type": "enabled", "budget_tokens": 10000}

                system_message = messages_[0]["content"]
                messages__ = messages_[1:]
                print(len(messages_))

                response = self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=16000,
                    thinking=thinking_config,
                    tools=anthropic_tools,
                    extra_headers=extra_headers,
                    system=system_message,
                    messages=messages__,
                    # temperature=temperature,
                )
                self.logger.warning(f"Anthropic Vertex query complete")
                # else:
                #     # Use litellm for other models
                #     kwargs = {
                #         "tool_choice": "none",
                #         "function_call": None,
                #     }
                #     if tools:
                #         kwargs = {}
                #     if "o3" not in self.llm_name and "o4" not in self.llm_name:
                #         kwargs["temperature"] = temperature
                #     response = litellm.completion(
                #         model=self.llm_name,
                #         tools=tools,
                #         messages=messages_,
                #         timeout=self.llm_timeout,
                #         api_base=self.llm_base_url,
                #         # max_tokens=3000,
                #         **kwargs,
                #     )
                #     self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e

        # End timer, calculate total execution time, and include in response
        exec_time = time.time() - start_time
        return response, exec_time

    def parse_response(self, response_text: str) -> Tuple[str, Action]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def custom_parser(self, response):
        thought = ""
        action = Action(function_name="", parameters={})

        # Handle Anthropic Vertex response
        if hasattr(response, "content") and isinstance(response.content, list):
            # Extract thought from text blocks
            thinking = ""
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text + "\n"
                elif block.type == "thinking":
                    thinking += block.thinking + "\n"
                elif block.type == "tool_use":
                    function_name = block.name
                    try:
                        # Try to parse input as JSON
                        if isinstance(block.input, str):
                            arguments = json.loads(block.input)
                        else:
                            arguments = block.input
                        action = Action(
                            function_name=function_name, parameters=arguments
                        )
                    except json.JSONDecodeError:
                        self.logger.error(
                            f"Failed to parse tool input as JSON: {block.input}"
                        )
                        action = Action(function_name=function_name, parameters={})
                    break  # Process only the first tool use

            # Format thought by combining thinking and text content
            thought = ""
            if thinking:
                thought += f"<think>\n{thinking}\n</think>\n\n"
            thought += text

        # Handle OpenAI-style response
        elif hasattr(response, "choices"):
            thought = response.choices[0].message.content
            if not thought:
                thought = ""
            try:
                function_name = response.choices[0].message.tool_calls[0].function.name
                parameters = json.loads(
                    response.choices[0].message.tool_calls[0].function.arguments
                )
                action = Action(function_name=function_name, parameters=parameters)
            except Exception as e:
                self.logger.error(f"Error parsing tool call in custom_parser: {e}")
                action = Action(function_name="", parameters={})

        return thought, action

    def run(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 32768,  # 32k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 1200,  # 20 minutes overall agent run limit
        max_llm_time: int = 120,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        condense_history: bool = True,
    ):

        # get the start time
        start_time = time.time()
        self.llm_timeout = max_llm_time

        # if self.llm_name is not gpt or sonnet, disable fn calling
        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            and "qwen" not in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        self.logger.info(f"User Prompt: {user_prompt}")

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []

        # agent loop
        while not done:
            # Prepare the agent's message history
            # self.logger.info(isinstance(steps_remaining, int))
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                # stepcount_message = f"Steps Remaining: {steps_remaining} Total Steps: {max_steps}"
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            # self.history[-1][
            #     "content"
            # ] += f"\n{stepcount_message}"  # postpend stepcount message
            # self.logger.info(stepcount_message)

            # Query the LLM
            messages = copy.deepcopy(self.history)
            try:
                response, llm_exec_time = self.model_query(
                    messages, temperature, condense_history=condense_history
                )
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                done = True
                exit_reason = "llm_query_error"
                break

            # Log total tokens in the response
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = -1
                if not condense_history:
                    total_tokens = self._count_tokens(messages)
                self.logger.warning(
                    "No token usage information available in the response."
                )

            # Parse the LLM response to get 'thought' and 'action'
            self.response = response  # for debugging

            if self.use_fn_calling:
                thought, action = self.custom_parser(response)
            else:
                assistant_message = response.choices[0].message.content
                self.logger.info(f"Assistant's message:\n{assistant_message}\n")
                thought, action = self.parse_response(assistant_message)

            action_str = action.to_xml_string()
            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                # env.runtime.commit_after_step(step_count)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count

            # Update history with assistant and tool responses
            if self.use_fn_calling:
                # if "vertex" in self.llm_name.lower() and self.anthropic_client:
                # Add assistant message to history
                ## TODO

                # get thinking blocks
                # get text blocks
                # get first tool use block
                # add to history

                thinking_blocks = [
                    block for block in response.content if block.type == "thinking"
                ]
                text_blocks = [
                    block for block in response.content if block.type == "text"
                ]
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ][:1]

                # Add assistant message to history with text content
                self.history.append(
                    {
                        "role": "assistant",
                        "content": thinking_blocks + text_blocks + tool_use_blocks,
                    }
                )

                # self.logger.warning(
                #     f"Added assistant message to history with content length: {len(text_content)}"
                # )

                # Add tool response to history if tool was used
                if tool_use_blocks:
                    tool_block = tool_use_blocks[0]
                    try:
                        self.history.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_block.id,
                                        "content": str(obs),
                                        # "tool_result": str(obs),
                                        # "name": tool_block.name,
                                        # "tool_call_id": tool_block.id
                                    }
                                ],
                            }
                        )
                        self.logger.warning(
                            f"Added tool response to history for tool: {tool_block.name}"
                        )
                    except Exception as e:
                        self.logger.error(f"Error adding tool response to history: {e}")
                        self.logger.warning("Falling back to user message")
                        self.history.append({"role": "user", "content": str(obs)})
                else:
                    self.logger.warning(
                        "No tool use blocks found, adding observation as user message"
                    )
                    self.history.append({"role": "user", "content": str(obs)})
                # else:
                #     # Handle OpenAI-style response
                #     assistant_response = response.choices[0].message.dict()
                #     if assistant_response.get("tool_calls", None):
                #         assistant_response["tool_calls"] = assistant_response["tool_calls"][
                #             :1
                #         ]  # only keep the first tool call
                #     self.history.append(assistant_response)
                #     # add tool response / user response to history
                #     try:
                #         function_name = (
                #             response.choices[0].message.tool_calls[0].function.name
                #         )
                #         function_id = response.choices[0].message.tool_calls[0].id
                #         self.history.append(
                #             {
                #                 "role": "tool",
                #                 "content": str(obs),
                #                 "name": function_name,
                #                 "tool_call_id": function_id,
                #             }
                #         )
                #         self.logger.warning("logging fn response as a tool call")
                #         self.logger.warning(
                #             f"number of fn calls: {len(response.choices[0].message.tool_calls)}"
                #         )
                #     except Exception as e:
                #         self.logger.error(f"Error logging tool response: {e}")
                #         self.logger.warning("fallback: logging fn response as a tool call")
                #         self.history.append({"role": "user", "content": str(obs)})
            else:
                self.logger.warning("logging fn response as a user message")
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": str(obs)})

            # Log the thought, action, and observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check if the agent has reached limits or done
            # check if agent has finished naturally i.e. the agent uses the finish tool
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            # check for token limit
            elif total_tokens >= max_token_limit:
                self.logger.info(
                    f"Agent reached max tokens: {max_token_limit}. Current token count: {total_tokens}. Exiting."
                )
                exit_reason = "token_limit"
                done = True
            # check for absolute step limit | note that the max steps is just indicative but the absolute step limit is the hard limit
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True

            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object and append to the list
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,  # also store the info to be safe
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(trajectory_step)
            time.sleep(15)  # Sleep for 60 seconds after each step

        # get the output patch
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD")
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD -- . ':(exclude)pyproject.toml'")
        # env.runtime.soft_git_reset()

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,  # reason for exiting. must be one of the [agent, max_step_limit, agent_max_step_limit, abs_step_limit, token_limit, traj_time_limit, llm_query_error]
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        return self.trajectory
