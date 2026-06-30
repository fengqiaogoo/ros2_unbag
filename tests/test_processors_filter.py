# MIT License

# Copyright (c) 2026 Institute for Automotive Engineering (ika), RWTH Aachen University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from ros2_unbag.core.processors.base import Processor
from ros2_unbag.core.processors.filter import keyframe_filter


@pytest.fixture(autouse=True)
def _reset_keyframe_state():
    """Reset persistent storage before each test."""
    keyframe_filter.persistent_storage.clear()


class FakeHeader:
    """Fake ROS message header with stamp."""
    class Stamp:
        def __init__(self, sec, nanosec):
            self.sec = sec
            self.nanosec = nanosec

    def __init__(self, sec=0, nanosec=0):
        self.stamp = self.Stamp(sec, nanosec)


class FakeImage:
    """Minimal fake sensor_msgs/Image."""
    def __init__(self, sec=0, nanosec=0):
        self.header = FakeHeader(sec, nanosec)


class FakeCompressedImage:
    """Minimal fake sensor_msgs/CompressedImage."""
    def __init__(self, sec=0, nanosec=0):
        self.header = FakeHeader(sec, nanosec)


class FakePointCloud2:
    """Minimal fake sensor_msgs/PointCloud2."""
    def __init__(self, sec=0, nanosec=0):
        self.header = FakeHeader(sec, nanosec)


class FakeOdometry:
    """Minimal fake nav_msgs/Odometry."""
    class Pose:
        class Pose:
            class Position:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            def __init__(self, x, y, z):
                self.position = self.Position(x, y, z)
        def __init__(self, x, y, z):
            self.pose = self.Pose(x, y, z)
    def __init__(self, x, y, z):
        self.pose = self.Pose(x, y, z)


class FakePoseStamped:
    """Minimal fake geometry_msgs/PoseStamped."""
    class Pose:
        class Position:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        def __init__(self, x, y, z):
            self.position = self.Position(x, y, z)
    def __init__(self, x, y, z):
        self.pose = self.Pose(x, y, z)


class TestKeyframeFilterFirstFrame:
    """Test that the first frame is always exported."""

    def test_first_frame_exported(self):
        msg = FakeImage(sec=1, nanosec=0)
        result = keyframe_filter(msg)
        assert result is msg

    def test_first_frame_exported_compressed(self):
        msg = FakeCompressedImage(sec=1, nanosec=0)
        result = keyframe_filter(msg)
        assert result is msg

    def test_first_frame_pointcloud(self):
        msg = FakePointCloud2(sec=1, nanosec=0)
        result = keyframe_filter(msg)
        assert result is msg

    def test_first_frame_with_odometry(self):
        odom = FakeOdometry(10.0, 20.0, 0.0)
        cross_topic_data = {"/odom": odom}
        msg = FakeImage(sec=1, nanosec=0)
        result = keyframe_filter(msg, cross_topic_data=cross_topic_data, pose_topic="/odom")
        assert result is msg


class TestKeyframeFilterTimeThreshold:
    """Test time-based keyframe extraction."""

    def test_time_threshold_triggers(self):
        # First frame at t=0
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1) is msg1

        # Second frame at t=1s (< 3s threshold) → skip
        msg2 = FakeImage(sec=1, nanosec=0)
        assert keyframe_filter(msg2) is None

        # Third frame at t=4s (>= 3s threshold) → export
        msg3 = FakeImage(sec=4, nanosec=0)
        assert keyframe_filter(msg3) is msg3

    def test_time_threshold_custom(self):
        keyframe_filter.persistent_storage.clear()

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, time_threshold=1.0) is msg1

        msg2 = FakeImage(sec=0, nanosec=500000000)  # 0.5s → skip
        assert keyframe_filter(msg2, time_threshold=1.0) is None

        msg3 = FakeImage(sec=1, nanosec=0)  # 1.0s → export
        assert keyframe_filter(msg3, time_threshold=1.0) is msg3

    def test_time_nanosec_overflow(self):
        keyframe_filter.persistent_storage.clear()

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1) is msg1

        # 2.5s in nanosecs: sec=2, nanosec=500_000_000
        msg2 = FakeImage(sec=2, nanosec=500000000)  # < 3s → skip
        assert keyframe_filter(msg2) is None

        # 3.1s: sec=3, nanosec=100_000_000 → should trigger
        msg3 = FakeImage(sec=3, nanosec=100000000)  # >= 3s → export
        assert keyframe_filter(msg3) is msg3


class TestKeyframeFilterDistanceThreshold:
    """Test distance-based keyframe extraction."""

    def test_distance_trigger_odometry(self):
        odom1 = FakeOdometry(0.0, 0.0, 0.0)
        odom2 = FakeOdometry(0.3, 0.0, 0.0)  # 0.3m (< 0.5m)
        odom3 = FakeOdometry(0.6, 0.0, 0.0)  # 0.6m (>= 0.5m)

        # Frame 1: t=0, pos=(0,0,0) → export
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/odom": odom1}) is msg1

        # Frame 2: t=0.1s, pos=(0.3,0,0) → distance 0.3m → skip
        msg2 = FakeImage(sec=0, nanosec=100000000)
        assert keyframe_filter(msg2, cross_topic_data={"/odom": odom2}) is None

        # Frame 3: t=0.2s, pos=(0.6,0,0) → distance 0.6m → export
        msg3 = FakeImage(sec=0, nanosec=200000000)
        assert keyframe_filter(msg3, cross_topic_data={"/odom": odom3}) is msg3

    def test_distance_euclidean_3d(self):
        odom1 = FakeOdometry(0.0, 0.0, 0.0)
        odom2 = FakeOdometry(0.3, 0.3, 0.3)  # sqrt(0.27) ≈ 0.52m

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/odom": odom1}) is msg1

        msg2 = FakeImage(sec=0, nanosec=100000000)
        assert keyframe_filter(msg2, cross_topic_data={"/odom": odom2}) is msg2  # >= 0.5m

    def test_distance_custom_threshold(self):
        keyframe_filter.persistent_storage.clear()

        odom1 = FakeOdometry(0.0, 0.0, 0.0)
        odom2 = FakeOdometry(0.3, 0.0, 0.0)

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/odom": odom1},
                               distance_threshold=1.0) is msg1

        msg2 = FakeImage(sec=0, nanosec=100000000)
        # 0.3m < 1.0m threshold AND time 0.1s < 3s → skip
        assert keyframe_filter(msg2, cross_topic_data={"/odom": odom2},
                               distance_threshold=1.0) is None

    def test_distance_pose_stamped(self):
        keyframe_filter.persistent_storage.clear()

        pose1 = FakePoseStamped(0.0, 0.0, 0.0)
        pose2 = FakePoseStamped(1.0, 0.0, 0.0)  # 1.0m

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/pose": pose1},
                               pose_topic="/pose") is msg1

        msg2 = FakeImage(sec=0, nanosec=100000000)
        assert keyframe_filter(msg2, cross_topic_data={"/pose": pose2},
                               pose_topic="/pose") is msg2


class TestKeyframeFilterNoOdometry:
    """Test behavior when no odometry data is available."""

    def test_time_only_without_odometry(self):
        # Without odometry, fallback to time-only mode
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1) is msg1

        msg2 = FakeImage(sec=1, nanosec=0)  # < 3s → skip
        assert keyframe_filter(msg2) is None

        msg3 = FakeImage(sec=3, nanosec=0)  # >= 3s → export
        assert keyframe_filter(msg3) is msg3

    def test_none_odometry_in_cross_topic(self):
        # Odom topic is in cross_topic_data but value is None
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/odom": None}) is msg1

        msg2 = FakeImage(sec=3, nanosec=0)  # time triggers
        assert keyframe_filter(msg2, cross_topic_data={"/odom": None}) is msg2

    def test_missing_odometry_topic(self):
        # Odom topic not in cross_topic_data at all
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/other": "data"},
                               pose_topic="/odom") is msg1

        msg2 = FakeImage(sec=3, nanosec=0)
        assert keyframe_filter(msg2, cross_topic_data={"/other": "data"},
                               pose_topic="/odom") is msg2


class TestKeyframeFilterCombined:
    """Test combined distance + time behavior."""

    def test_stationary_triggers_time(self):
        """Vehicle is stationary (odometry not changing), time should trigger."""
        odom = FakeOdometry(0.0, 0.0, 0.0)

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/odom": odom}) is msg1

        # Same position at 1s, 2s → skip both
        msg2 = FakeImage(sec=1, nanosec=0)
        assert keyframe_filter(msg2, cross_topic_data={"/odom": odom}) is None

        msg3 = FakeImage(sec=2, nanosec=0)
        assert keyframe_filter(msg3, cross_topic_data={"/odom": odom}) is None

        # Same position at 3s → time triggers
        msg4 = FakeImage(sec=3, nanosec=0)
        assert keyframe_filter(msg4, cross_topic_data={"/odom": odom}) is msg4

    def test_rapid_movement_triggers_distance(self):
        """Vehicle moves fast; distance triggers before time threshold."""

        # First frame
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={
            "/odom": FakeOdometry(0.0, 0.0, 0.0)
        }) is msg1

        # 0.1s later, moved 0.6m → distance triggers (time 0.1s < 3s)
        msg2 = FakeImage(sec=0, nanosec=100000000)
        assert keyframe_filter(msg2, cross_topic_data={
            "/odom": FakeOdometry(0.6, 0.0, 0.0)
        }) is msg2

    def test_reset_after_distance_export(self):
        """After a distance-triggered export, the position reference resets."""
        odom1 = FakeOdometry(0.0, 0.0, 0.0)
        odom2 = FakeOdometry(0.6, 0.0, 0.0)  # 0.6m → triggers
        odom3 = FakeOdometry(0.8, 0.0, 0.0)  # only 0.2m from last keyframe

        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1, cross_topic_data={"/odom": odom1}) is msg1

        msg2 = FakeImage(sec=0, nanosec=100000000)
        assert keyframe_filter(msg2, cross_topic_data={"/odom": odom2}) is msg2

        # Now at odom3, only 0.2m from last keyframe → skip
        msg3 = FakeImage(sec=0, nanosec=200000000)
        assert keyframe_filter(msg3, cross_topic_data={"/odom": odom3}) is None

    def test_reset_after_time_export(self):
        """After a time-triggered export, the timer resets."""
        msg1 = FakeImage(sec=0, nanosec=0)
        assert keyframe_filter(msg1) is msg1

        msg2 = FakeImage(sec=3, nanosec=0)
        assert keyframe_filter(msg2) is msg2

        # Only 1s after last export → skip
        msg3 = FakeImage(sec=4, nanosec=0)
        assert keyframe_filter(msg3) is None


class TestKeyframeFilterValidation:
    """Test input validation."""

    def test_invalid_thresholds(self):
        msg = FakeImage(sec=0, nanosec=0)
        with pytest.raises(ValueError):
            keyframe_filter(msg, distance_threshold="not_a_number")

        keyframe_filter.persistent_storage.clear()
        with pytest.raises(ValueError):
            keyframe_filter(msg, time_threshold="invalid")


class TestProcessorRegistration:
    """Test that the keyframe_filter is properly registered as a Processor."""

    @classmethod
    def setup_class(cls):
        # Re-import filter module to re-register processors in case the
        # registry was cleared by another test module's setup_function.
        import importlib
        import ros2_unbag.core.processors.filter as fmod
        importlib.reload(fmod)

    def test_registered_as_filter(self):
        formats = Processor.get_formats("sensor_msgs/msg/Image")
        assert "keyframe_filter" in formats

        formats_pc = Processor.get_formats("sensor_msgs/msg/PointCloud2")
        assert "keyframe_filter" in formats_pc

        handler = Processor.get_handler("sensor_msgs/msg/Image", "keyframe_filter")
        assert handler is not None
        assert getattr(handler, 'is_filter', False) is True

    def test_cross_topic_deps(self):
        handler = Processor.get_handler("sensor_msgs/msg/Image", "keyframe_filter")
        proc_inst = getattr(handler, '_processor_instance', None)
        assert proc_inst is not None

        deps = proc_inst.get_cross_topic_deps({"pose_topic": "/my_odom"})
        assert deps == ["/my_odom"]

        deps_empty = proc_inst.get_cross_topic_deps({})
        assert deps_empty == []

    def test_cross_topic_deps_no_pose_topic_arg(self):
        handler = Processor.get_handler("sensor_msgs/msg/Image", "keyframe_filter")
        proc_inst = getattr(handler, '_processor_instance', None)

        deps = proc_inst.get_cross_topic_deps({"other_arg": "value"})
        assert deps == []
