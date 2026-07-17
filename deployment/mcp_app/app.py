from tools.mcp_server import mcp

if __name__ == "__main__":
    # Databricks Apps provides the port via $DATABRICKS_APP_PORT (default 8000).
    mcp.run(transport="streamable-http")
