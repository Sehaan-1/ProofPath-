"""Local ProofPath MCP server package (Slice 1.7)."""

from proofpath.mcp_server.server import (
    SERVER_NAME,
    SERVER_TITLE,
    SERVER_VERSION,
    FastMCP,
    MCPServer,
    MCPServerRuntime,
    MCPTransport,
    SessionContextFactory,
    build_arg_parser,
    create_mcp_server,
    create_streamable_http_app,
    main,
    mcp,
    run_mcp_server,
)

__all__ = [
    "SERVER_NAME",
    "SERVER_TITLE",
    "SERVER_VERSION",
    "FastMCP",
    "MCPServer",
    "MCPServerRuntime",
    "MCPTransport",
    "SessionContextFactory",
    "build_arg_parser",
    "create_mcp_server",
    "create_streamable_http_app",
    "main",
    "mcp",
    "run_mcp_server",
]
