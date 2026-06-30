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

import math
import logging

from ros2_unbag.core.processors.base import Processor

logger = logging.getLogger(__name__)


@Processor(
    ["sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image",
     "sensor_msgs/msg/PointCloud2"],
    ["keyframe_filter"],
    is_filter=True,
    cross_topic_arg_keys=["pose_topic"],
)
def keyframe_filter(msg, cross_topic_data=None, pose_topic: str = "/odom",
                    distance_threshold: float = 0.5, time_threshold: float = 3.0):
    """
    Filter messages to extract keyframes based on distance and time thresholds.

    A message is selected as a keyframe when EITHER of these conditions is met:
    1. The Euclidean distance traveled since the last keyframe is >= distance_threshold (meters).
    2. The time elapsed since the last keyframe is >= time_threshold (seconds).
    The very first message is always selected as a keyframe.

    Distance is computed from the latest message on the specified pose_topic
    (nav_msgs/Odometry, geometry_msgs/PoseStamped, or geometry_msgs/TransformStamped).

    Args:
        msg: CompressedImage or Image ROS 2 message instance.
        cross_topic_data (dict): Mapping of topic names to their latest messages.
        pose_topic (str): Topic name that provides pose/odometry data for distance calculation.
        distance_threshold (float): Minimum distance (meters) between keyframes.
        time_threshold (float): Minimum time (seconds) between keyframes when stationary.

    Returns:
        The original message if it qualifies as a keyframe, or None to skip.

    Raises:
        TypeError: If the pose message type is unsupported.
    """
    # --- validate thresholds ---
    try:
        distance_threshold = float(distance_threshold)
        time_threshold = float(time_threshold)
    except (TypeError, ValueError):
        raise ValueError(
            f"distance_threshold ({distance_threshold}) and time_threshold "
            f"({time_threshold}) must be numeric."
        )

    # --- persistent state ---
    state = keyframe_filter.persistent_storage

    # --- current timestamp (nanoseconds) ---
    current_time = _extract_timestamp_ns(msg)

    # --- current position from cross-topic data ---
    current_position = _extract_position(cross_topic_data, pose_topic)

    # --- first message: always export ---
    if 'last_export_time' not in state:
        state['last_export_time'] = current_time
        state['last_export_position'] = current_position
        return msg

    time_diff_s = (current_time - state['last_export_time']) / 1e9

    # --- check time threshold ---
    if time_diff_s >= time_threshold:
        state['last_export_time'] = current_time
        if current_position is not None:
            state['last_export_position'] = current_position
        return msg

    # --- check distance threshold ---
    if current_position is not None and state.get('last_export_position') is not None:
        dx = current_position[0] - state['last_export_position'][0]
        dy = current_position[1] - state['last_export_position'][1]
        dz = current_position[2] - state['last_export_position'][2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance >= distance_threshold:
            state['last_export_time'] = current_time
            state['last_export_position'] = current_position
            return msg

    # --- neither condition met: skip ---
    return None


def _extract_timestamp_ns(msg):
    """Extract timestamp in nanoseconds from a ROS message header."""
    try:
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        return int(sec) * 1_000_000_000 + int(nanosec)
    except AttributeError:
        try:
            sec = msg.stamp.sec
            nanosec = msg.stamp.nanosec
            return int(sec) * 1_000_000_000 + int(nanosec)
        except AttributeError:
            logger.warning("Message has no valid timestamp; using 0")
            return 0


def _extract_position(cross_topic_data, pose_topic):
    """
    Extract (x, y, z) position from the latest message on pose_topic in cross_topic_data.

    Supports nav_msgs/Odometry, geometry_msgs/PoseStamped, and
    geometry_msgs/TransformStamped.
    """
    if not cross_topic_data or pose_topic not in cross_topic_data:
        return None

    pose_msg = cross_topic_data[pose_topic]
    if pose_msg is None:
        return None

    # Try to identify message type by duck-typing (avoid hard imports that may fail)
    msg_type = type(pose_msg).__name__

    try:
        # nav_msgs/msg/Odometry
        if hasattr(pose_msg, 'pose') and hasattr(pose_msg.pose, 'pose'):
            pos = pose_msg.pose.pose.position
            return (pos.x, pos.y, pos.z)
        # geometry_msgs/msg/PoseStamped
        elif hasattr(pose_msg, 'pose') and hasattr(pose_msg.pose, 'position'):
            pos = pose_msg.pose.position
            return (pos.x, pos.y, pos.z)
        # geometry_msgs/msg/TransformStamped
        elif hasattr(pose_msg, 'transform') and hasattr(pose_msg.transform, 'translation'):
            t = pose_msg.transform.translation
            return (t.x, t.y, t.z)
        else:
            logger.warning(
                f"Unsupported pose message type '{msg_type}' on topic '{pose_topic}'. "
                f"Expected Odometry, PoseStamped, or TransformStamped. "
                f"Distance-based filtering disabled."
            )
            return None
    except Exception as e:
        logger.warning(f"Failed to extract position from pose message on '{pose_topic}': {e}")
        return None
