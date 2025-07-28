from fastapi.responses import JSONResponse
from constants.http_enum import HttpStatusCode


def http_response_success(status_code: HttpStatusCode, message: str, data=None):
    """Generate a JSON response for a successful operation."""
    response = {
        "success": True,
        "message": message,
    }

    if data is not None:
        response["data"] = data

    return JSONResponse(status_code=status_code.value, content=response)


def http_response_error(status_code: HttpStatusCode, message: str, error_details=None):
    """Generate a JSON response for a failed operation."""
    response = {
        "success": False,
        "message": message,
    }

    if error_details is not None:
        response["data"] = error_details

    return JSONResponse(status_code=status_code.value, content=response)
