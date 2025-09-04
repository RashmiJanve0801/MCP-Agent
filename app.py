import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import FunctionCallTermination, TextMentionTermination

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")

async def config():
    params = StdioServerParams(
        command=r"C:\Program Files\nodejs\npx.cmd",
        args=['-y','mcp-remote','https://mcp.notion.com/mcp'],
        env={
            'NOTION_API_KEY': NOTION_API_KEY
        },
        read_timeout_seconds=120
    )

    open_router_model_client = OpenAIChatCompletionClient(
        base_url = "https://openrouter.ai/api/v1",
        model = "deepseek/deepseek-chat-v3-0324:free",
        api_key = os.getenv("OPENROUTER_API_KEY"),
        model_info = {
            "family":"deepseek",
            "vision":True,
            "function_calling":True,
            "json_output":False
        }
    )

    model_client = OpenAIChatCompletionClient(
        model = "gemini-2.5-pro",
        api_key=os.getenv("GOOGLE_API_KEY"),
         model_info={
                        "vision":True,
                        "function_calling":True,
                        "json_output":False,
                        "structured_output":True,
                        "family":"google",
                        "mode":"chat"
                    }
    )

    mcp_tools = await mcp_server_tools(server_params=params)

    agent = AssistantAgent(
        name="Notion_Agent",
        system_message="You are a helpful assistant that can search and summarize content from the user's Notion workspace and also list what is asked.Try to assume the tool and call the same and get the answer. Say TERMINATE when you are done with the task.",
        model_client=open_router_model_client,
        tools=mcp_tools,
        reflect_on_tool_use=True
    )

    team = RoundRobinGroupChat(
        participants=[agent],
        max_turns=5,
        termination_condition=TextMentionTermination('TERMINATE')
    )

    return team

async def orchestrate(team, task):
    async for msg in team.run_stream(task=task):
        yield(msg)

async def main():
    team = await config()
    task = "Create a new page titled 'Page from MCP Notion'"

    async for msg in orchestrate(team, task):
        print(msg)

if __name__ == "__main__":
    asyncio.run(main())
