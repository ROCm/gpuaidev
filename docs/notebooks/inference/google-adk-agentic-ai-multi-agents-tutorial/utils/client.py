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

import httpx
import base64
from typing import Any, AsyncIterable
from utils.a2a_types import (
    AgentCard,
    SendTaskRequest,
    SendTaskResponse,
    JSONRPCRequest,
    A2AClientHTTPError,
    A2AClientJSONError,
    SendTaskStreamingResponse,
)
import json


class A2AClient:
    def __init__(self, agent_card: AgentCard, auth: str, agent_url: str):
        # The URL accessed here should be the same as the one provided in the agent card
        # However, in this demo we are using the URL provided in the key arguments
        self.url = agent_url
        # self.url = agent_card.url
        self.auth_header = None

        if agent_card.authentication:
            if len(agent_card.authentication.schemes) > 1:
                raise ValueError(
                    "Only one A2A client authentication scheme is supported for now"
                )
            elif len(agent_card.authentication.schemes) == 1:
                if agent_card.authentication.schemes[0].lower() == "bearer":
                    self.auth_header = f"Bearer {auth}"
                elif agent_card.authentication.schemes[0].lower() == "basic":
                    # Encode auth string to base64 for Basic authentication
                    encoded_auth = base64.b64encode(auth.encode()).decode()
                    self.auth_header = f"Basic {encoded_auth}"
                else:
                    raise ValueError("Unsupported authentication scheme")

    async def send_task(self, payload: dict[str, Any]) -> SendTaskResponse:
        request = SendTaskRequest(params=payload)
        return SendTaskResponse(**await self._send_request(request))

    async def send_task_streaming(
        self, payload: dict[str, Any]
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        raise NotImplementedError("Streaming is not supported for now")

    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                # Image generation could take time, adding timeout
                print(f"Send Remote Agent Task Request: {request.model_dump()}")
                print("=" * 100)
                request_kwargs = {
                    "url": self.url,
                    "json": request.model_dump(),
                    "timeout": 30,
                }
                if self.auth_header:
                    request_kwargs["headers"] = {"Authorization": self.auth_header}

                response = await client.post(**request_kwargs)
                response.raise_for_status()
                print(f"Send Remote Agent Task Response: {response.json()}")
                print("=" * 100)
                return response.json()
            except httpx.HTTPStatusError as e:
                raise A2AClientHTTPError(e.response.status_code, str(e)) from e
            except json.JSONDecodeError as e:
                raise A2AClientJSONError(str(e)) from e
