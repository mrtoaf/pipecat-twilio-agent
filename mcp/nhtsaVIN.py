from typing import Any, Optional
import httpx
import re
from mcp.server.fastmcp import FastMCP

# Init FastMCP server
mcp = FastMCP("nhtsa-vin-number")

# Constants
NHTSA_API_BASE = "https://vpic.nhtsa.dot.gov/api"
USER_AGENT = "vin-decoder/1.0"

async def make_nhtsa_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NHTSA API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_vehicle_info(data: dict) -> str:
    """Format vehicle information into a readable string, filtering out empty fields."""
    if not data or "Results" not in data or not data["Results"]:
        return "No vehicle information found for this VIN."

    results = data["Results"][0]

    important_fields = [
        "VIN", "Make", "Model", "ModelYear", "Series", "Trim",
        "Manufacturer", "BodyClass", "VehicleType", "DriveType",
        "EngineConfiguration", "EngineCylinders", "EngineHP", "Displacement",
        "FuelTypePrimary", "Doors", "Seats", "SeatRows", "ErrorText"
    ]

    vehicle_info = []
    for field in important_fields:
        # Check if the field exists and has a meaningful value
        value = results.get(field)
        if value is not None and str(value).strip() not in ("", "/", "Not Applicable"):
             # Specific check for ErrorText being '0' which means no error
            if field == "ErrorText" and str(value) == "0":
                continue # Skip displaying 'ErrorText: 0'
            vehicle_info.append(f"{field}: {value}")

    if not vehicle_info:
         return "Found basic record, but no specific details available for this VIN."

    return "\n".join(vehicle_info)


@mcp.tool()
async def decode_vin(vin: str = None, modelyear: Optional[str] = None) -> str:
    """Decode a Vehicle Identification Number (VIN) using NHTSA's API.

    Args:
        vin: The Vehicle Identification Number to decode.
        modelyear: Optional model year if known (improves accuracy).
    """
    if not vin:
        return "Please provide a VIN number to decode."

    # Clean the VIN - remove spaces, hyphens, common punctuation
    clean_vin = re.sub(r'[\s\-.,]', '', vin)
    # Ensure it's uppercase
    clean_vin = clean_vin.upper()
    # Basic check for likely invalid characters (VINs use letters and numbers only)
    if not re.fullmatch(r'[A-HJ-NPR-Z0-9]+', clean_vin):
        return f"The provided VIN '{vin}' contains invalid characters. Please provide a valid VIN."

    # Proceed with lookup only if confirmed
    if len(clean_vin) != 17:
        note = f" (Note: Standard VINs are 17 characters, this one has {len(clean_vin)}.)"
    else:
        note = ""

    confirmation_msg = f"Looking up confirmed VIN: {clean_vin}{note}"

    url = f"{NHTSA_API_BASE}/vehicles/DecodeVinValues/{clean_vin}?format=json"

    if modelyear:
        # Clean model year just in case
        clean_modelyear = re.sub(r'\D', '', str(modelyear)) # Keep only digits
        if clean_modelyear:
            url += f"&modelyear={clean_modelyear}"

    data = await make_nhtsa_request(url)

    if not data:
        return f"{confirmation_msg}\n\nUnable to fetch vehicle information from NHTSA for VIN {clean_vin}."

    vehicle_info = format_vehicle_info(data)

    header = "Vehicle Information:"
    return f"{confirmation_msg}\n\n{header}\n\n{vehicle_info}"


if __name__ == "__main__":
    # Ensure the directory exists if running directly
    import os
    if not os.path.exists('mcp'):
        os.makedirs('mcp', exist_ok=True)
        # Create a dummy file if needed for relative path testing
        with open('mcp/__init__.py', 'w') as f:
            pass

    mcp.run(transport='stdio')