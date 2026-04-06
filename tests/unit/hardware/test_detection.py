"""Unit tests for platform detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.hardware._platforms._detection import Platform, detect_platform


@pytest.mark.unit
class TestPlatformEnum:
    """Test Platform enum members."""

    def test_platform_enum_has_qcs6490(self) -> None:
        """Test Platform.QCS6490 exists."""
        assert hasattr(Platform, "QCS6490")
        assert Platform.QCS6490 is not None

    def test_platform_enum_has_x86_64(self) -> None:
        """Test Platform.X86_64 exists."""
        assert hasattr(Platform, "X86_64")
        assert Platform.X86_64 is not None

    def test_platform_enum_all_members(self) -> None:
        """Test Platform enum has all expected members."""
        members = list(Platform)
        assert len(members) == 3
        assert Platform.QCS6490 in members
        assert Platform.X86_64 in members
        assert Platform.MACOS_ARM64 in members


@pytest.mark.unit
class TestDetectPlatform:
    """Test platform detection."""

    @pytest.fixture(autouse=True)
    def _clear_detect_cache(self) -> None:
        """Clear the detect_platform cache so each test can patch inputs safely."""
        detect_platform.cache_clear()

    def test_detect_x86_64_on_laptop(self) -> None:
        """Test detect_platform returns X86_64 on this laptop."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platforms._detection._QCOM_SOC_NAME_FILE", mock_path),
            patch("platform.machine", return_value="x86_64"),
        ):
            platform = detect_platform()
            assert platform == Platform.X86_64

    def test_detect_x86_64_amd64_alias(self) -> None:
        """Test detect_platform recognizes amd64 as x86_64."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platforms._detection._QCOM_SOC_NAME_FILE", mock_path),
            patch("platform.machine", return_value="amd64"),
        ):
            platform = detect_platform()
            assert platform == Platform.X86_64

    def test_detect_qcs6490(self) -> None:
        """Test detect_platform recognizes QCS6490 from sysfs."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "QCS6490\n"
        with (
            patch("moment_to_action.hardware._platforms._detection._QCOM_SOC_NAME_FILE", mock_path),
            patch("platform.machine", return_value="aarch64"),
        ):
            platform = detect_platform()
            assert platform == Platform.QCS6490

    def test_detect_unrecognized_soc_raises(self) -> None:
        """Test detect_platform raises RuntimeError for unrecognized platform."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = "__NOT_A_KNOWN_PLATFORM__\n"
        with (
            patch("moment_to_action.hardware._platforms._detection._QCOM_SOC_NAME_FILE", mock_path),
            patch("platform.machine", return_value="aarch64"),
            patch("platform.system", return_value="Linux"),
        ):
            with pytest.raises(RuntimeError, match="Unrecognised platform"):
                detect_platform()

    def test_detect_unrecognized_platform_raises(self) -> None:
        """Test detect_platform raises RuntimeError for unrecognized platform."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platforms._detection._QCOM_SOC_NAME_FILE", mock_path),
            patch("platform.machine", return_value="arm64"),
            patch("platform.system", return_value="Linux"),
        ):
            with pytest.raises(RuntimeError, match="Unrecognised platform"):
                detect_platform()

    def test_detect_macos_arm64(self) -> None:
        """Test detect_platform recognizes Apple Silicon macOS."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with (
            patch("moment_to_action.hardware._platforms._detection._QCOM_SOC_NAME_FILE", mock_path),
            patch("platform.machine", return_value="arm64"),
            patch("platform.system", return_value="Darwin"),
        ):
            platform = detect_platform()
            assert platform == Platform.MACOS_ARM64
