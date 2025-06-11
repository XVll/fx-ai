"""
Tests for EpisodeManagerException class.
Tests exception creation, reason handling, and message formatting.
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.episode_manager import EpisodeManagerException, EpisodeTerminationReason


class TestEpisodeManagerException:
    """Test suite for EpisodeManagerException class."""

    def test_exception_creation_with_reason_only(self):
        """Test creating exception with reason only (no custom message)."""
        exc = EpisodeManagerException(EpisodeTerminationReason.NO_MORE_DAYS)
        
        assert exc.reason == EpisodeTerminationReason.NO_MORE_DAYS
        assert exc.message == "Episode manager terminated: no_more_days"
        assert str(exc) == "Episode manager terminated: no_more_days"

    def test_exception_creation_with_custom_message(self):
        """Test creating exception with custom message."""
        custom_message = "Custom error message for testing"
        exc = EpisodeManagerException(EpisodeTerminationReason.PRELOAD_FAILED, custom_message)
        
        assert exc.reason == EpisodeTerminationReason.PRELOAD_FAILED
        assert exc.message == custom_message
        assert str(exc) == custom_message

    def test_all_termination_reasons_can_be_used(self):
        """Test that all termination reasons can be used to create exceptions."""
        reasons = [
            EpisodeTerminationReason.CYCLE_LIMIT_REACHED,
            EpisodeTerminationReason.EPISODE_LIMIT_REACHED,
            EpisodeTerminationReason.UPDATE_LIMIT_REACHED,
            EpisodeTerminationReason.NO_MORE_RESET_POINTS,
            EpisodeTerminationReason.NO_MORE_DAYS,
            EpisodeTerminationReason.DATE_RANGE_EXHAUSTED,
            EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET,
            EpisodeTerminationReason.PRELOAD_FAILED,
        ]
        
        for reason in reasons:
            exc = EpisodeManagerException(reason)
            assert exc.reason == reason
            assert f"Episode manager terminated: {reason.value}" in str(exc)

    def test_exception_inherits_from_exception(self):
        """Test that EpisodeManagerException inherits from Exception."""
        exc = EpisodeManagerException(EpisodeTerminationReason.NO_MORE_DAYS)
        assert isinstance(exc, Exception)

    def test_exception_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught properly."""
        with pytest.raises(EpisodeManagerException) as exc_info:
            raise EpisodeManagerException(EpisodeTerminationReason.NO_MORE_RESET_POINTS)
        
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_RESET_POINTS

    def test_exception_with_none_message_uses_default(self):
        """Test that passing None as message uses default message."""
        exc = EpisodeManagerException(EpisodeTerminationReason.DATE_RANGE_EXHAUSTED, None)
        
        assert exc.reason == EpisodeTerminationReason.DATE_RANGE_EXHAUSTED
        assert exc.message == "Episode manager terminated: date_range_exhausted"
        assert str(exc) == "Episode manager terminated: date_range_exhausted"

    def test_exception_with_empty_string_message(self):
        """Test that passing empty string as message uses default."""
        exc = EpisodeManagerException(EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET, "")
        
        assert exc.reason == EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET
        # Empty string should trigger default message
        assert exc.message == "Episode manager terminated: quality_criteria_not_met"
        assert str(exc) == "Episode manager terminated: quality_criteria_not_met"

    def test_exception_reason_attribute_accessible(self):
        """Test that reason attribute is accessible after creation."""
        reason = EpisodeTerminationReason.PRELOAD_FAILED
        exc = EpisodeManagerException(reason, "Test message")
        
        # Should be able to access reason directly
        assert hasattr(exc, 'reason')
        assert exc.reason == reason
        assert exc.reason.value == "preload_failed"

    def test_exception_message_attribute_accessible(self):
        """Test that message attribute is accessible after creation."""
        message = "Custom test message"
        exc = EpisodeManagerException(EpisodeTerminationReason.NO_MORE_DAYS, message)
        
        # Should be able to access message directly
        assert hasattr(exc, 'message')
        assert exc.message == message

    def test_exception_string_representation_formats_correctly(self):
        """Test that string representation is formatted correctly."""
        test_cases = [
            (EpisodeTerminationReason.NO_MORE_DAYS, None, "Episode manager terminated: no_more_days"),
            (EpisodeTerminationReason.PRELOAD_FAILED, "Custom message", "Custom message"),
            (EpisodeTerminationReason.NO_MORE_RESET_POINTS, "Reset points exhausted", "Reset points exhausted"),
        ]
        
        for reason, message, expected_str in test_cases:
            exc = EpisodeManagerException(reason, message)
            assert str(exc) == expected_str

    def test_exception_with_multiline_message(self):
        """Test that multiline messages are handled correctly."""
        multiline_message = "Line 1\nLine 2\nLine 3"
        exc = EpisodeManagerException(EpisodeTerminationReason.DATE_RANGE_EXHAUSTED, multiline_message)
        
        assert exc.message == multiline_message
        assert str(exc) == multiline_message

    def test_exception_reason_enum_properties(self):
        """Test that reason enum properties are preserved."""
        exc = EpisodeManagerException(EpisodeTerminationReason.QUALITY_CRITERIA_NOT_MET)
        
        # Should be able to access enum properties
        assert exc.reason.name == "QUALITY_CRITERIA_NOT_MET"
        assert exc.reason.value == "quality_criteria_not_met"

    def test_exception_string_representation_varies_by_reason_and_message(self):
        """Test that string representation varies based on reason and message."""
        exc1 = EpisodeManagerException(EpisodeTerminationReason.NO_MORE_DAYS, "Test message")
        exc2 = EpisodeManagerException(EpisodeTerminationReason.NO_MORE_DAYS, "Test message")
        exc3 = EpisodeManagerException(EpisodeTerminationReason.NO_MORE_DAYS, "Different message")
        exc4 = EpisodeManagerException(EpisodeTerminationReason.PRELOAD_FAILED, "Different reason")
        
        # Same reason and message should have same string representation
        assert str(exc1) == str(exc2)
        assert str(exc1) != str(exc3)
        assert str(exc1) != str(exc4)

    def test_exception_can_be_used_in_try_except_blocks(self):
        """Test that exception works properly in try-except blocks."""
        def raise_episode_exception():
            raise EpisodeManagerException(EpisodeTerminationReason.NO_MORE_RESET_POINTS, "Test exception")
        
        # Should be able to catch specifically
        with pytest.raises(EpisodeManagerException) as exc_info:
            raise_episode_exception()
        
        assert exc_info.value.reason == EpisodeTerminationReason.NO_MORE_RESET_POINTS
        assert "Test exception" in str(exc_info.value)
        
        # Should also be catchable as general Exception
        try:
            raise_episode_exception()
        except Exception as e:
            assert isinstance(e, EpisodeManagerException)
            assert e.reason == EpisodeTerminationReason.NO_MORE_RESET_POINTS