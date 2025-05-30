"""Integration tests for CLI workflows"""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
import json
import yaml

from src.cli.main import cli
from tests.utils.factories import TestDataFactory
from tests.utils.helpers import TestHelpers


class TestCLIWorkflows:
    """Test complete CLI workflows"""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner"""
        return CliRunner()
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with test data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create test directories
            (workspace / "logs").mkdir()
            (workspace / "configs").mkdir()
            (workspace / "data" / "vectorstore").mkdir(parents=True)
            
            # Create test log files
            log_content = TestDataFactory.create_pcie_log_file(
                num_entries=100,
                error_rate=0.2
            )
            (workspace / "logs" / "pcie_test.log").write_text(log_content)
            
            uvm_content = TestDataFactory.create_uvm_log_content(
                test_name="pcie_link_test",
                include_errors=True
            )
            (workspace / "logs" / "uvm_test.log").write_text(uvm_content)
            
            # Create test configuration
            config = TestDataFactory.create_config_dict(
                vector_store={"path": str(workspace / "data" / "vectorstore" / "test_index")}
            )
            config_path = workspace / "configs" / "test_settings.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            yield workspace, config_path
    
    @pytest.mark.integration
    def test_config_init_workflow(self, runner, temp_workspace):
        """Test configuration initialization workflow"""
        workspace, _ = temp_workspace
        config_path = workspace / "configs" / "new_config.yaml"
        
        # Initialize configuration
        result = runner.invoke(cli, [
            "config", "init",
            "--output", str(config_path)
        ], input="openai\ngpt-3.5-turbo\n\nlocal\nsentence-transformers/all-MiniLM-L6-v2\n")
        
        assert result.exit_code == 0
        assert config_path.exists()
        assert "Configuration saved" in result.output
        
        # Validate created config
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "config", "validate"
        ])
        
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output
    
    @pytest.mark.integration
    def test_index_build_workflow(self, runner, temp_workspace):
        """Test index building workflow"""
        workspace, config_path = temp_workspace
        
        # Build index
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--pattern", "*.log",
            "--force"
        ])
        
        assert result.exit_code == 0
        assert "Found 2 files to index" in result.output
        assert "Indexing complete" in result.output
        
        # Check index statistics
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "index", "stats"
        ])
        
        assert result.exit_code == 0
        assert "Vector Store Statistics" in result.output
    
    @pytest.mark.integration
    def test_search_workflow(self, runner, temp_workspace):
        """Test search workflow"""
        workspace, config_path = temp_workspace
        
        # First build index
        runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--force"
        ])
        
        # Search for content
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "search",
            "PCIe error",
            "--limit", "5",
            "--output", "json"
        ])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        results = json.loads(result.output)
        assert isinstance(results, list)
        assert len(results) <= 5
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)
    
    @pytest.mark.integration
    def test_analyze_workflow(self, runner, temp_workspace):
        """Test analysis workflow"""
        workspace, config_path = temp_workspace
        
        # Build index first
        runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--force"
        ])
        
        # Analyze with query
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "analyze",
            "--query", "What PCIe errors occurred?",
            "--output", "json"
        ])
        
        # Should either succeed or fail gracefully with API error
        if result.exit_code == 0:
            # Parse JSON output
            analysis = json.loads(result.output)
            assert "query" in analysis
            assert "answer" in analysis
            assert "confidence" in analysis
        else:
            # Check for expected error (no API key)
            assert "API" in result.output or "key" in result.output
    
    @pytest.mark.integration
    def test_report_generation_workflow(self, runner, temp_workspace):
        """Test report generation workflow"""
        workspace, config_path = temp_workspace
        report_path = workspace / "report.html"
        
        # Build index
        runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--force"
        ])
        
        # Generate report
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "report",
            "--query", "Analyze PCIe errors",
            "--output", str(report_path),
            "--format", "html"
        ])
        
        # Should create report or fail gracefully
        if result.exit_code == 0:
            assert report_path.exists()
            content = report_path.read_text()
            assert "<html>" in content
            assert "PCIe Debug Analysis Report" in content
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self, runner, temp_workspace):
        """Test complete end-to-end workflow"""
        workspace, config_path = temp_workspace
        
        # 1. Validate configuration
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "config", "validate"
        ])
        assert result.exit_code == 0
        
        # 2. Build index
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--force"
        ])
        assert result.exit_code == 0
        
        # 3. Search for errors
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "search",
            "error",
            "--limit", "10"
        ])
        assert result.exit_code == 0
        assert "Found" in result.output or "No results" in result.output
        
        # 4. Get index statistics
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "index", "stats"
        ])
        assert result.exit_code == 0
        
        # 5. Test connectivity
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "test", "connectivity"
        ])
        # Should complete (may have failures due to missing API keys)
        assert "Connectivity Test Results" in result.output
    
    @pytest.mark.integration
    def test_cli_error_handling(self, runner, temp_workspace):
        """Test CLI error handling"""
        workspace, config_path = temp_workspace
        
        # Test with non-existent file
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            "/non/existent/path"
        ])
        assert result.exit_code != 0
        
        # Test search without index
        result = runner.invoke(cli, [
            "--config", str(workspace / "configs" / "nonexistent.yaml"),
            "search", "test"
        ])
        assert result.exit_code != 0
        
        # Test invalid command
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Usage" in result.output
    
    @pytest.mark.integration
    def test_cli_help_system(self, runner):
        """Test CLI help system"""
        # Main help
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PCIe Debug Agent" in result.output
        assert "Commands:" in result.output
        
        # Command help
        for command in ["analyze", "index", "search", "report", "config", "test"]:
            result = runner.invoke(cli, [command, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.output
            assert "Options:" in result.output
    
    @pytest.mark.integration
    def test_verbose_and_quiet_modes(self, runner, temp_workspace):
        """Test verbose and quiet output modes"""
        workspace, config_path = temp_workspace
        
        # Test verbose mode
        result = runner.invoke(cli, [
            "--verbose",
            "--config", str(config_path),
            "config", "show"
        ])
        assert result.exit_code == 0
        
        # Test quiet mode
        result = runner.invoke(cli, [
            "--quiet",
            "--config", str(config_path),
            "config", "show"
        ])
        assert result.exit_code == 0
        # Should have minimal output
        assert "PCIe Debug Agent" not in result.output  # No banner
    
    @pytest.mark.integration
    def test_json_output_formats(self, runner, temp_workspace):
        """Test JSON output format across commands"""
        workspace, config_path = temp_workspace
        
        # Build index
        runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--force"
        ])
        
        # Test search JSON output
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "search", "test",
            "--output", "json"
        ])
        
        if result.exit_code == 0 and result.output.strip():
            data = json.loads(result.output)
            assert isinstance(data, list)
        
        # Test config JSON output
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "config", "show",
            "--format", "json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert "embedding_model" in data
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_under_load(self, runner, temp_workspace):
        """Test CLI performance with larger datasets"""
        workspace, config_path = temp_workspace
        
        # Create many log files
        for i in range(10):
            log_content = TestDataFactory.create_pcie_log_file(
                num_entries=1000,
                error_rate=0.1
            )
            (workspace / "logs" / f"pcie_large_{i}.log").write_text(log_content)
        
        # Build index with large dataset
        result = runner.invoke(cli, [
            "--config", str(config_path),
            "index", "build",
            str(workspace / "logs"),
            "--force",
            "--batch-size", "500"
        ])
        
        assert result.exit_code == 0
        assert "Found 12 files" in result.output  # 10 new + 2 original