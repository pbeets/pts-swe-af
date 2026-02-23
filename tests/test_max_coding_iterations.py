"""Test max_coding_iterations configuration changes."""

import pytest
from swe_af.execution.schemas import ExecutionConfig, BuildConfig


class TestMaxCodingIterationsDefault:
    """Test that ExecutionConfig default max_coding_iterations is 6."""

    def test_execution_config_default_is_6(self):
        """Verify ExecutionConfig has default max_coding_iterations of 6."""
        config = ExecutionConfig()
        assert config.max_coding_iterations == 6

    def test_build_config_default_is_6(self):
        """Verify BuildConfig has default max_coding_iterations of 6."""
        config = BuildConfig()
        assert config.max_coding_iterations == 6


class TestMaxCodingIterationsOverride:
    """Test that max_coding_iterations can be overridden."""

    def test_execution_config_override_with_8(self):
        """Verify ExecutionConfig max_coding_iterations can be set to 8."""
        config = ExecutionConfig(max_coding_iterations=8)
        assert config.max_coding_iterations == 8

    def test_execution_config_override_with_5(self):
        """Verify existing configs with max_coding_iterations=5 work (AC3)."""
        config = ExecutionConfig(max_coding_iterations=5)
        assert config.max_coding_iterations == 5

    def test_build_config_override_with_8(self):
        """Verify BuildConfig max_coding_iterations can be set to 8."""
        config = BuildConfig(max_coding_iterations=8)
        assert config.max_coding_iterations == 8

    def test_build_config_override_with_5(self):
        """Verify existing configs with max_coding_iterations=5 work (AC3)."""
        config = BuildConfig(max_coding_iterations=5)
        assert config.max_coding_iterations == 5


class TestMaxCodingIterationsTransfer:
    """Test that max_coding_iterations transfers from BuildConfig to ExecutionConfig."""

    def test_build_to_execution_config_dict_includes_max_coding_iterations(self):
        """Verify max_coding_iterations is included in to_execution_config_dict()."""
        build_config = BuildConfig(max_coding_iterations=6)
        exec_config_dict = build_config.to_execution_config_dict()
        assert "max_coding_iterations" in exec_config_dict
        assert exec_config_dict["max_coding_iterations"] == 6

    def test_build_to_execution_config_dict_with_override(self):
        """Verify overridden max_coding_iterations transfers correctly."""
        build_config = BuildConfig(max_coding_iterations=8)
        exec_config_dict = build_config.to_execution_config_dict()
        assert exec_config_dict["max_coding_iterations"] == 8


class TestMaxCodingIterationsCeiling:
    """Test that the iteration ceiling is enforced in execution.

    Note: The actual enforcement happens in coding_loop.py via:
        max_iterations = config.max_coding_iterations
    This test validates the config schema provides the correct value.
    """

    def test_default_ceiling_is_6(self):
        """Verify the default ceiling for coding iterations is 6 (AC4)."""
        config = ExecutionConfig()
        # The coding loop will use this value to limit iterations
        assert config.max_coding_iterations == 6

    def test_ceiling_can_be_configured(self):
        """Verify the ceiling can be adjusted via configuration."""
        # Test that a higher ceiling can be set if needed
        config = ExecutionConfig(max_coding_iterations=10)
        assert config.max_coding_iterations == 10

        # Test that a lower ceiling can be set if needed
        config = ExecutionConfig(max_coding_iterations=3)
        assert config.max_coding_iterations == 3


class TestBackwardCompatibility:
    """Test backward compatibility with existing configurations."""

    def test_execution_config_with_explicit_5_still_works(self):
        """Verify that old configs explicitly setting 5 continue to work."""
        # Simulates an existing config that was using the old default
        config = ExecutionConfig(
            max_coding_iterations=5,
            runtime="claude_code",
        )
        assert config.max_coding_iterations == 5

    def test_build_config_with_explicit_5_still_works(self):
        """Verify that old BuildConfig with explicit 5 continues to work."""
        config = BuildConfig(
            max_coding_iterations=5,
            runtime="claude_code",
        )
        assert config.max_coding_iterations == 5
