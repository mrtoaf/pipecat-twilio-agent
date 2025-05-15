import shutil
from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient
from mcp import StdioServerParameters

from bot.config import BotConfig

async def register_mcp_clients(llm: OpenAILLMService, cfg: BotConfig) -> ToolsSchema:
    """Initializes MCP clients, registers their tools, and returns a merged schema."""
    uvx_path = shutil.which("uvx")
    if not uvx_path:
        logger.error("uvx command not found in PATH. Please install uv.")
        # Consider raising an exception here or handling it more gracefully
        # depending on whether MCP tools are critical or optional.
        return ToolsSchema(standard_tools=[]) # Return empty schema if uvx is not found

    time_server_params = StdioServerParameters(
        command=uvx_path,
        args=["mcp-server-time", "--local-timezone", "America/New_York"]
    )

    vin_server_params = StdioServerParameters(
        command="python3",
        args=["./mcp/nhtsaVIN.py"]
    )

    time_mcp_client = MCPClient(server_params=time_server_params)
    logger.info("Time MCP client initialized.")

    vin_mcp_client = MCPClient(server_params=vin_server_params)
    logger.info("VIN MCP client initialized.")

    all_mcp_tools = []

    try:
        time_schema = await time_mcp_client.register_tools(llm)
        logger.info(f"Registered tools from time server: {time_schema}")
        if time_schema:
            all_mcp_tools.extend(time_schema.standard_tools)
    except Exception as e:
        logger.error(f"Failed to register tools from time server: {e}")

    try:
        vin_schema = await vin_mcp_client.register_tools(llm)
        logger.info(f"Registered tools from VIN server: {vin_schema}")
        if vin_schema:
            all_mcp_tools.extend(vin_schema.standard_tools)
    except Exception as e:
        logger.error(f"Failed to register tools from VIN server: {e}")
    
    merged_tools_schema = ToolsSchema(standard_tools=all_mcp_tools)
    logger.info(f"Merged MCP tools schema: {merged_tools_schema}")

    return merged_tools_schema 