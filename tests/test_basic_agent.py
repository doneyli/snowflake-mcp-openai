"""
Tests for Basic Agent

Unit tests for the basic Snowflake MCP agent functionality.
"""
import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.basic_agent import main


class TestBasicAgent:
    """Test cases for the basic agent"""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'SNOWFLAKE_ACCOUNT_IDENTIFIER': 'test-account',
            'SNOWFLAKE_USERNAME': 'test-user',
            'SNOWFLAKE_PAT': 'test-pat-token'
        }):
            yield
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Mock MCP server for testing"""
        server_mock = AsyncMock()
        server_mock.list_tools.return_value = [
            Mock(name="mcp_mcp-server-snowflake_cortex_analyst_sales_orders"),
            Mock(name="mcp_mcp-server-snowflake_cortex_complete"),
            Mock(name="mcp_mcp-server-snowflake_cortex_search_leases")
        ]
        return server_mock
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent for testing"""
        agent_mock = Mock()
        agent_mock.name = "test-agent"
        return agent_mock
    
    @pytest.fixture  
    def mock_runner_result(self):
        """Mock runner result for testing"""
        result_mock = Mock()
        result_mock.final_output = "Test analysis result from Snowflake"
        return result_mock
    
    def test_environment_validation_missing_openai_key(self):
        """Test that missing OpenAI API key raises appropriate error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
                asyncio.run(main())
    
    def test_environment_validation_missing_snowflake_creds(self, mock_env_vars):
        """Test that missing Snowflake credentials raise appropriate error"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            with pytest.raises(RuntimeError, match="Missing Snowflake credentials"):
                asyncio.run(main())
    
    @patch('agents.basic_agent.Runner')
    @patch('agents.basic_agent.Agent')
    @patch('agents.basic_agent.MCPServerStdio')
    async def test_successful_agent_execution(
        self, 
        mock_mcp_server_class,
        mock_agent_class,
        mock_runner_class,
        mock_env_vars,
        mock_mcp_server,
        mock_agent,
        mock_runner_result
    ):
        """Test successful agent execution with mocked dependencies"""
        
        # Setup mocks
        mock_mcp_server_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_server)
        mock_mcp_server_class.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_agent_class.return_value = mock_agent
        mock_runner_class.run = AsyncMock(return_value=mock_runner_result)
        
        # Execute main function
        await main()
        
        # Verify MCP server was created with correct parameters
        mock_mcp_server_class.assert_called_once()
        call_args = mock_mcp_server_class.call_args
        
        assert call_args[1]['params']['command'] == 'uvx'
        assert '--from' in call_args[1]['params']['args']
        assert 'git+https://github.com/Snowflake-Labs/mcp' in call_args[1]['params']['args']
        assert call_args[1]['name'] == 'snowflake-mcp-server'
        assert call_args[1]['client_session_timeout_seconds'] == 60
        
        # Verify agent was created
        mock_agent_class.assert_called_once()
        agent_call_args = mock_agent_class.call_args
        assert agent_call_args[1]['name'] == 'snowflake-agent'
        assert 'Snowflake Cortex AI tools' in agent_call_args[1]['instructions']
        
        # Verify tools were listed
        mock_mcp_server.list_tools.assert_called_once()
        
        # Verify runner was called
        mock_runner_class.run.assert_called_once()
        run_call_args = mock_runner_class.run.call_args
        assert run_call_args[0][0] == mock_agent
        assert isinstance(run_call_args[0][1], str)  # Query string
    
    @patch('agents.basic_agent.Runner')
    @patch('agents.basic_agent.Agent')
    @patch('agents.basic_agent.MCPServerStdio')
    async def test_agent_execution_error_handling(
        self,
        mock_mcp_server_class,
        mock_agent_class, 
        mock_runner_class,
        mock_env_vars,
        mock_mcp_server,
        mock_agent
    ):
        """Test error handling during agent execution"""
        
        # Setup mocks with error
        mock_mcp_server_class.return_value.__aenter__ = AsyncMock(return_value=mock_mcp_server)
        mock_mcp_server_class.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_agent_class.return_value = mock_agent
        mock_runner_class.run = AsyncMock(side_effect=Exception("Test execution error"))
        
        # Execute should not raise exception (error is caught and printed)
        await main()
        
        # Verify runner was called and error was handled
        mock_runner_class.run.assert_called_once()


@pytest.mark.asyncio
class TestBasicAgentIntegration:
    """Integration tests for the basic agent (require actual credentials)"""
    
    @pytest.mark.skipif(
        not all([
            os.getenv('OPENAI_API_KEY'),
            os.getenv('SNOWFLAKE_ACCOUNT_IDENTIFIER'),
            os.getenv('SNOWFLAKE_USERNAME'),
            os.getenv('SNOWFLAKE_PAT')
        ]),
        reason="Integration tests require actual credentials"
    )
    async def test_real_agent_execution(self):
        """Test with real credentials (only runs if environment is properly configured)"""
        # This test will only run if all environment variables are set
        # Useful for CI/CD integration testing
        try:
            await main()
            # If we get here without exception, the integration worked
            assert True
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 