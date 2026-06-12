"""
Dedicated Netlify Function for Inngest.

Uses CommHandler directly (no FastAPI/Mangum) so the function URL
/.netlify/functions/inngest is the Inngest serve endpoint with no
path-routing ambiguity.
"""
import asyncio
import base64
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _here,
    os.path.normpath(os.path.join(_here, "..", "..")),
    "/var/task",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv()

from inngest._internal import comm_lib, server_lib
from main import inngest_client, rag_ingest_pdf, rag_query_pdf_ai

_SERVE_PATH = "/.netlify/functions/inngest"

_comm = comm_lib.CommHandler(
    client=inngest_client,
    framework=server_lib.Framework.FAST_API,
    functions=[rag_ingest_pdf, rag_query_pdf_ai],
    streaming=None,
)


def handler(event, context):
    method = (event.get("httpMethod") or "GET").upper()
    headers = dict(event.get("headers") or {})

    raw_body = event.get("body") or b""
    if isinstance(raw_body, str):
        raw_body = raw_body.encode("utf-8")
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body)

    query_params = dict(event.get("queryStringParameters") or {})

    host = headers.get("host", "")
    proto = headers.get("x-forwarded-proto", "https")
    qs = ("?" + "&".join(f"{k}={v}" for k, v in query_params.items())) if query_params else ""
    request_url = f"{proto}://{host}{_SERVE_PATH}{qs}"

    req = comm_lib.CommRequest(
        body=raw_body,
        headers=headers,
        public_path=None,
        query_params=query_params,
        raw_request=event,
        request_url=request_url,
        serve_origin=None,
        serve_path=_SERVE_PATH,
    )

    if method == "GET":
        resp = _comm.get_sync(req)
    elif method == "POST":
        resp = asyncio.run(_comm.post(req))
    elif method == "PUT":
        resp = asyncio.run(_comm.put(req))
    else:
        return {"statusCode": 405, "body": "Method Not Allowed"}

    body_out = (
        json.dumps(resp.body)
        if isinstance(resp.body, (dict, list))
        else (resp.body or "")
    )
    return {
        "statusCode": resp.status_code,
        "body": body_out,
        "headers": dict(resp.headers or {}),
    }
