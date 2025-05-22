from pipecat_flows import FlowArgs, FlowResult, FlowManager
from loguru import logger
import re

async def store_name(args: FlowArgs) -> FlowResult:
    """Store the user's name from flow arguments."""
    name = args.get("name")
    logger.info(f"Storing name: {name}")
    # The name will be in the LLM context from the function call result.
    # If needed in flow_manager.state for other purposes, it could be set here too,
    # possibly via a transition_callback if strict separation is desired.
    return {"status": "success", "name": name}

async def choose_time_handler(args: FlowArgs) -> FlowResult:
    """User chose to get the current time."""
    logger.debug("User chose to get time.")
    return {"status": "success"}

async def choose_vin_handler(args: FlowArgs) -> FlowResult:
    """User chose to decode a VIN."""
    logger.debug("User chose to decode VIN.")
    return {"status": "success"}

async def collect_vin_handler(args: FlowArgs) -> FlowResult:
    """Collects the VIN from the arguments, cleans it, validates length, and returns it."""
    raw_vin = args.get("vin", "")
    # Remove any non-alphanumeric chars (spaces, hyphens, commas, etc.) and uppercase
    cleaned_vin = re.sub(r'[^A-Za-z0-9]', '', raw_vin).upper()
    
    # Validate the cleaned VIN: 17 characters, valid VIN characters (A-H, J-N, P, R-Z, 0-9)
    if not re.fullmatch(r'[A-HJ-NPR-Z0-9]{17}', cleaned_vin):
        logger.warning(f"collect_vin_handler: Raw VIN '{raw_vin}' cleaned to '{cleaned_vin}' is invalid.")
        return {
            "status": "error", 
            "message": "The V I N you provided was not a valid 17-character V I N. Please try again, saying or entering the full V I N clearly."
        }
    
    logger.info(f"VIN collected, cleaned, and validated: {cleaned_vin} (from raw: '{raw_vin}')")
    return {"status": "success", "vin": cleaned_vin}

async def store_vin_in_state_callback(standard_args: dict, handler_result: FlowResult, flow_manager: FlowManager):
    """Transition callback to store the VIN (from collect_vin_handler result) into flow_manager.state."""
    if handler_result and handler_result.get("status") == "success":
        vin_to_store = handler_result.get("vin")
        if vin_to_store:
            logger.info(f"Storing VIN '{vin_to_store}' into flow_manager.state via transition_callback.")
            await flow_manager.state.set("vin", vin_to_store)
        else:
            logger.warning("store_vin_in_state_callback: VIN missing in handler_result.")
    elif handler_result:
        logger.warning(f"store_vin_in_state_callback: Handler result was not success: {handler_result}")
    else:
        logger.warning("store_vin_in_state_callback: No handler_result provided.")

async def confirm_vin_handler(args: FlowArgs) -> FlowResult:
    """User confirmed the VIN is correct."""
    logger.debug("User confirmed VIN.")
    return {"status": "success"}

async def reject_vin_handler(args: FlowArgs) -> FlowResult:
    """User stated the VIN is incorrect."""
    logger.debug("User rejected VIN. Returning to VIN collection.")
    return {"status": "success"} 