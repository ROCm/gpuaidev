"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from utils.a2a_types import (
    A2ARequest,
    JSONRPCResponse,
    InvalidRequestError,
    JSONParseError,
    GetTaskRequest,
    CancelTaskRequest,
    SendTaskRequest,
    SetTaskPushNotificationRequest,
    GetTaskPushNotificationRequest,
    InternalError,
    AgentCard,
    TaskResubscriptionRequest,
    SendTaskStreamingRequest,
)
from pydantic import ValidationError
import json
from typing import AsyncIterable, Any
from utils.task_manager_base import TaskManager

import logging
import base64

logger = logging.getLogger(__name__)


class A2AServer:
    def __init__(
        self,
        host="0.0.0.0",
        port=5000,
        endpoint="/",
        agent_card: AgentCard = None,
        task_manager: TaskManager = None,
        api_key: str | None = None,
        auth_username: str | None = None,
        auth_password: str | None = None,
    ):
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.task_manager = task_manager
        self.agent_card = agent_card
        self.api_key = api_key
        self.auth_username = auth_username
        self.auth_password = auth_password
        self.app = Starlette()
        self.app.add_route(self.endpoint, self._process_request, methods=["POST"])
        self.app.add_route(
            "/.well-known/agent.json", self._get_agent_card, methods=["GET"]
        )

        if len(self.agent_card.authentication.schemes) > 1:
            raise ValueError("Only one authentication scheme is supported for now")

        if self.agent_card.authentication.schemes[0].lower() == "bearer":
            self.auth_scheme = "bearer"

            if self.api_key is None:
                raise ValueError(
                    "Authentication scheme is bearer but api_key is not defined"
                )
        elif self.agent_card.authentication.schemes[0].lower() == "basic":
            self.auth_scheme = "basic"

            if self.auth_username is None or self.auth_password is None:
                raise ValueError(
                    "Authentication scheme is basic but auth_username and auth_password are not defined"
                )
        else:
            raise ValueError("Unsupported authentication scheme")

    def start(self):
        if self.agent_card is None:
            raise ValueError("agent_card is not defined")

        if self.task_manager is None:
            raise ValueError("request_handler is not defined")

        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    def _get_agent_card(self, request: Request) -> JSONResponse:
        return JSONResponse(self.agent_card.model_dump(exclude_none=True))

    def verify_bearer_token(self, token):
        """Verify the provided bearer token against the expected token."""
        # Simple token comparison for demonstration
        # In a real application, you might want to use a more secure verification method
        return token == self.api_key

    def verify_basic_auth(self, username, password):
        """Verify the provided basic auth credentials against the expected credentials."""
        return username == self.auth_username and password == self.auth_password

    async def verify_auth_header(self, request):
        """Verify the authorization header based on the configured auth scheme."""
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return False, "Authorization header is missing"

        parts = auth_header.split()
        if len(parts) != 2:
            return False, "Invalid Authorization header format"

        auth_type, credentials = parts
        auth_type = auth_type.lower()

        # Handle different authentication schemes
        if self.auth_scheme == "bearer" and auth_type == "bearer":
            if not self.api_key:
                return True, None  # Skip auth if no API key is set

            if not self.verify_bearer_token(credentials):
                return False, "Invalid bearer token"
            return True, None

        elif self.auth_scheme == "basic" and auth_type == "basic":
            if not self.auth_username or not self.auth_password:
                return True, None  # Skip auth if credentials not set

            try:
                decoded = base64.b64decode(credentials).decode("utf-8")
                username, password = decoded.split(":", 1)
                if not self.verify_basic_auth(username, password):
                    return False, "Invalid credentials"
                return True, None
            except Exception as e:
                logger.error(f"Error decoding basic auth: {e}")
                return False, "Invalid basic auth format"
        else:
            return (
                False,
                f"Authentication scheme mismatch. Expected {self.auth_scheme}, got {auth_type}",
            )

    async def _process_request(self, request: Request):
        # Check authentication based on configured auth scheme
        is_valid, error_message = await self.verify_auth_header(request)
        if not is_valid:
            return JSONResponse({"error": error_message}, status_code=401)

        try:
            body = await request.json()
            json_rpc_request = A2ARequest.validate_python(body)
            print(json_rpc_request)

            if isinstance(json_rpc_request, GetTaskRequest):
                result = await self.task_manager.on_get_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskRequest):
                result = await self.task_manager.on_send_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskStreamingRequest):
                result = await self.task_manager.on_send_task_subscribe(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, CancelTaskRequest):
                result = await self.task_manager.on_cancel_task(json_rpc_request)
            elif isinstance(json_rpc_request, SetTaskPushNotificationRequest):
                result = await self.task_manager.on_set_task_push_notification(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, GetTaskPushNotificationRequest):
                result = await self.task_manager.on_get_task_push_notification(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, TaskResubscriptionRequest):
                result = await self.task_manager.on_resubscribe_to_task(
                    json_rpc_request
                )
            else:
                logger.warning(f"Unexpected request type: {type(json_rpc_request)}")
                raise ValueError(f"Unexpected request type: {type(request)}")

            return self._create_response(result)

        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> JSONResponse:
        if isinstance(e, json.decoder.JSONDecodeError):
            json_rpc_error = JSONParseError()
        elif isinstance(e, ValidationError):
            json_rpc_error = InvalidRequestError(data=json.loads(e.json()))
        else:
            logger.error(f"Unhandled exception: {e}")
            json_rpc_error = InternalError()

        response = JSONRPCResponse(id=None, error=json_rpc_error)
        return JSONResponse(response.model_dump(exclude_none=True), status_code=400)

    def _create_response(self, result: Any) -> JSONResponse | EventSourceResponse:
        if isinstance(result, AsyncIterable):

            async def event_generator(result) -> AsyncIterable[dict[str, str]]:
                async for item in result:
                    yield {"data": item.model_dump_json(exclude_none=True)}

            return EventSourceResponse(event_generator(result))
        elif isinstance(result, JSONRPCResponse):
            return JSONResponse(result.model_dump(exclude_none=True))
        else:
            logger.error(f"Unexpected result type: {type(result)}")
            raise ValueError(f"Unexpected result type: {type(result)}")
