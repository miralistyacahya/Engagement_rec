from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from constants.http_enum import HttpStatusCode
from utils.response import http_response_error

from routes import predict_router

app = FastAPI(title="Engagement Recognition API")

app.include_router(predict_router)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return http_response_error(exc.status_code, exc.detail)

@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    return http_response_error(HttpStatusCode.BadRequest, "Invalid request body", exc.errors())

@app.exception_handler(Exception)
def exception_handler(request: Request, exc: Exception):
    return http_response_error(HttpStatusCode.InternalServerError, "Internal server error")
