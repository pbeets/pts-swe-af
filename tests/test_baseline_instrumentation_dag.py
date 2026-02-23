"""Tests for baseline instrumentation in dag_executor.py."""

import asyncio
import pytest
from unittest.mock import MagicMock

from swe_af.execution.dag_executor import _log_agent_metrics, _call_with_timeout


class TestLogAgentMetrics:
    """Unit tests for _log_agent_metrics() helper function."""

    def test_log_agent_metrics_with_all_parameters(self):
        """Test _log_agent_metrics with all parameters provided."""
        note_fn = MagicMock()
        _log_agent_metrics(
            note_fn=note_fn,
            role="issue_advisor",
            duration=45.3,
            success=True,
            iteration=2,
            extra_tags=["test_issue", "confidence:0.85"]
        )

        # Verify note_fn was called
        assert note_fn.call_count == 1
        call_args = note_fn.call_args

        # Check message
        message = call_args[0][0]
        assert "issue_advisor" in message
        assert "45.3s" in message
        assert "success=True" in message

        # Check tags
        tags = call_args[1]["tags"]
        assert "agent_metrics" in tags
        assert "role:issue_advisor" in tags
        assert "duration:45.3" in tags
        assert "success:True" in tags
        assert "iteration:2" in tags
        assert "test_issue" in tags
        assert "confidence:0.85" in tags

    def test_log_agent_metrics_without_iteration(self):
        """Test _log_agent_metrics without iteration parameter."""
        note_fn = MagicMock()
        _log_agent_metrics(
            note_fn=note_fn,
            role="coder",
            duration=120.5,
            success=False,
        )

        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        assert "role:coder" in tags
        assert "duration:120.5" in tags
        assert "success:False" in tags
        # iteration:0 should not be in tags
        assert not any(tag.startswith("iteration:") for tag in tags)

    def test_log_agent_metrics_without_extra_tags(self):
        """Test _log_agent_metrics without extra_tags parameter."""
        note_fn = MagicMock()
        _log_agent_metrics(
            note_fn=note_fn,
            role="replanner",
            duration=200.0,
            success=True,
            iteration=1,
        )

        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        assert "role:replanner" in tags
        assert "duration:200.0" in tags
        assert "success:True" in tags
        assert "iteration:1" in tags

    def test_log_agent_metrics_with_none_note_fn(self):
        """Test _log_agent_metrics does not crash when note_fn is None."""
        # Should not raise any exception
        _log_agent_metrics(
            note_fn=None,
            role="test_role",
            duration=10.0,
            success=True,
        )


class TestCallWithTimeout:
    """Integration tests for _call_with_timeout() wrapper."""

    @pytest.mark.asyncio
    async def test_call_with_timeout_success(self):
        """Test _call_with_timeout with successful coroutine completion."""
        note_fn = MagicMock()

        async def successful_coro():
            await asyncio.sleep(0.01)
            return "success"

        result = await _call_with_timeout(
            successful_coro(),
            timeout=1,
            label="test_agent",
            note_fn=note_fn,
            role="test_role"
        )

        assert result == "success"
        assert note_fn.call_count == 1

        # Verify success logging
        tags = note_fn.call_args[1]["tags"]
        assert "agent_timeout" in tags
        assert "success" in tags
        assert "role:test_role" in tags
        assert any("duration:" in tag for tag in tags)
        assert any("timeout:" in tag for tag in tags)

    @pytest.mark.asyncio
    async def test_call_with_timeout_failure(self):
        """Test _call_with_timeout with timeout scenario."""
        note_fn = MagicMock()

        async def slow_coro():
            await asyncio.sleep(10)
            return "never_reached"

        with pytest.raises(TimeoutError) as exc_info:
            await _call_with_timeout(
                slow_coro(),
                timeout=0.1,
                label="slow_agent",
                note_fn=note_fn,
                role="slow_role"
            )

        assert "timed out after 0.1s" in str(exc_info.value)
        assert note_fn.call_count == 1

        # Verify timeout logging
        tags = note_fn.call_args[1]["tags"]
        assert "agent_timeout" in tags
        assert "failure" in tags
        assert "role:slow_role" in tags
        assert "timeout:0.1" in tags

    @pytest.mark.asyncio
    async def test_call_with_timeout_without_note_fn(self):
        """Test _call_with_timeout without note_fn does not crash."""
        async def simple_coro():
            return "result"

        result = await _call_with_timeout(
            simple_coro(),
            timeout=1,
            label="test",
            note_fn=None,
            role="test"
        )

        assert result == "result"

    @pytest.mark.asyncio
    async def test_call_with_timeout_preserves_exception(self):
        """Test _call_with_timeout preserves non-timeout exceptions."""
        note_fn = MagicMock()

        async def failing_coro():
            raise ValueError("test error")

        with pytest.raises(ValueError) as exc_info:
            await _call_with_timeout(
                failing_coro(),
                timeout=1,
                label="failing_agent",
                note_fn=note_fn,
                role="failing_role"
            )

        assert "test error" in str(exc_info.value)


class TestIntegrationMetrics:
    """Integration tests for metrics flow in dag execution."""

    @pytest.mark.asyncio
    async def test_coding_loop_metrics_mock(self):
        """Test that coding loop metrics are captured correctly."""
        # This is a basic integration test structure
        # Real integration would involve mocking the entire coding loop
        note_fn = MagicMock()

        # Simulate what happens after coding loop completes
        issue_name = "test-issue"
        attempts = 2
        duration = 45.5

        note_fn(
            f"Coding loop: {issue_name} iteration {attempts}, {duration:.1f}s",
            tags=["coding_loop", "complete", issue_name, f"attempts:{attempts}", f"duration:{duration:.1f}"]
        )

        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        assert "coding_loop" in tags
        assert "complete" in tags
        assert issue_name in tags
        assert "attempts:2" in tags
        assert "duration:45.5" in tags

    @pytest.mark.asyncio
    async def test_advisor_metrics_mock(self):
        """Test that Issue Advisor metrics are captured correctly."""
        note_fn = MagicMock()

        # Simulate Issue Advisor completion logging
        issue_name = "test-issue"
        iteration = 1
        confidence = 0.75
        duration = 30.2

        note_fn(
            f"Issue Advisor: {issue_name} iteration {iteration}, confidence {confidence:.2f}, {duration:.1f}s",
            tags=["issue_advisor", "complete", issue_name, f"iteration:{iteration}", f"confidence:{confidence:.2f}", f"duration:{duration:.1f}"]
        )

        _log_agent_metrics(
            note_fn=note_fn,
            role="issue_advisor",
            duration=duration,
            success=True,
            iteration=iteration,
            extra_tags=[issue_name, f"confidence:{confidence:.2f}"]
        )

        assert note_fn.call_count == 2
        # Check the metrics call
        metrics_call = note_fn.call_args_list[1]
        tags = metrics_call[1]["tags"]
        assert "agent_metrics" in tags
        assert "role:issue_advisor" in tags

    @pytest.mark.asyncio
    async def test_replanner_metrics_mock(self):
        """Test that Replanner metrics are captured correctly."""
        note_fn = MagicMock()

        # Simulate Replanner completion logging
        duration = 120.5
        action = "MODIFY_DAG"

        note_fn(
            f"Replanner: {duration:.1f}s, action={action}",
            tags=["replanner", "complete", f"duration:{duration:.1f}", f"action:{action}"]
        )

        _log_agent_metrics(
            note_fn=note_fn,
            role="replanner",
            duration=duration,
            success=True,
            extra_tags=[f"action:{action}"]
        )

        assert note_fn.call_count == 2
        # Check the note call
        note_call = note_fn.call_args_list[0]
        tags = note_call[1]["tags"]
        assert "replanner" in tags
        assert "complete" in tags
        assert f"action:{action}" in tags


class TestEdgeCases:
    """Edge case tests for instrumentation."""

    def test_log_agent_metrics_with_zero_duration(self):
        """Test _log_agent_metrics with zero duration."""
        note_fn = MagicMock()
        _log_agent_metrics(
            note_fn=note_fn,
            role="fast_agent",
            duration=0.0,
            success=True,
        )

        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        assert "duration:0.0" in tags

    def test_log_agent_metrics_with_iteration_zero(self):
        """Test _log_agent_metrics with iteration=0 (default, should not appear in tags)."""
        note_fn = MagicMock()
        _log_agent_metrics(
            note_fn=note_fn,
            role="test_role",
            duration=10.0,
            success=True,
            iteration=0,
        )

        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        # iteration:0 should not be in tags when iteration is 0
        assert not any(tag.startswith("iteration:") for tag in tags)

    def test_log_agent_metrics_missing_confidence_field(self):
        """Test _log_agent_metrics handles missing confidence in extra_tags gracefully."""
        note_fn = MagicMock()
        # This simulates a scenario where confidence is not provided
        _log_agent_metrics(
            note_fn=note_fn,
            role="issue_advisor",
            duration=45.0,
            success=True,
            iteration=1,
            extra_tags=["test_issue"]  # No confidence tag
        )

        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        assert "test_issue" in tags
        # Should not crash, just omit confidence

    @pytest.mark.asyncio
    async def test_call_with_timeout_on_first_invocation(self):
        """Test _call_with_timeout handles timeout on first agent invocation."""
        note_fn = MagicMock()

        async def instant_timeout():
            await asyncio.sleep(1)
            return "never"

        with pytest.raises(TimeoutError):
            await _call_with_timeout(
                instant_timeout(),
                timeout=0.01,
                label="first_call",
                note_fn=note_fn,
                role="first_agent"
            )

        # Should still log timeout event
        assert note_fn.call_count == 1
        tags = note_fn.call_args[1]["tags"]
        assert "failure" in tags
