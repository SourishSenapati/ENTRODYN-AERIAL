"""
Unit tests for Swarm Consensus Logic.
Uses mocked ROS2 nodes to verify Byzantine Fault Tolerance.
"""


import sys
import os
import unittest
from unittest.mock import MagicMock

# 1. Mock module 'rclpy' FIRST (Simulating Linux ROS2 environment on Windows)
# This allows 'from entrodyn_ros.swarm_controller' to import 'rclpy' as a mock
mock_rclpy = MagicMock()
mock_node_module = MagicMock()


class MockNode:
    """Mock ROS2 Node for testing without ROS installation."""

    def __init__(self, node_name):
        self.node_name = node_name
        self.get_logger = MagicMock()
        self.create_publisher = MagicMock()
        self.create_subscription = MagicMock()


sys.modules['rclpy'] = mock_rclpy
sys.modules['rclpy.node'] = mock_node_module
sys.modules['rclpy.node'].Node = MockNode

# 2. Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

# 3. Import the SwarmController (after mocking and path setup)
if 'entrodyn_ros.swarm_controller' not in sys.modules:
    # pylint: disable=import-error
    from entrodyn_ros.swarm_controller import SwarmController
else:
    SwarmController = sys.modules['entrodyn_ros.swarm_controller'].SwarmController


class TestByzantineFaultTolerance(unittest.TestCase):
    """
    Test Suite for PBFT Logic (Practical Byzantine Fault Tolerance).
    Verifies consensus mechanisms and resilience against compromised nodes.
    """

    def setUp(self):
        self.controller = SwarmController()
        # Mock 5 peers (Total 6 drones including self)
        # Quorum needed: 0.67 * 5 = 3.35 -> 4 votes
        self.peers = [MagicMock() for _ in range(5)]
        self.controller.peer_drones = self.peers

    def test_consensus_reached(self):
        """Test that consensus is reached when all peers agree"""
        print("\nTesting: Normal Operation (Consensus Reached)")
        proposed_velocity = [1.0, 0.5, 0.0]

        # All peers agree
        for p in self.peers:
            p.validate_vector.return_value = True

        result = self.controller.check_consensus(proposed_velocity)
        self.assertTrue(result)
        print(">> Consensus: ACHIEVED (Swarm Moves)")

    def test_byzantine_attack_mitigated(self):
        """Test that consensus rejects rogue commands"""
        print("\nTesting: Cyber-Attack / Byzantine Fault")
        proposed_velocity = [999.0, 999.0, 999.0]  # Rogue command

        # 3 peers reject (Honest), 2 peers accept (Compromised/Faulty)
        # Votes = 2. Quorum = 4.
        self.peers[0].validate_vector.return_value = False
        self.peers[1].validate_vector.return_value = False
        self.peers[2].validate_vector.return_value = False
        self.peers[3].validate_vector.return_value = True  # Hacked
        self.peers[4].validate_vector.return_value = True  # Hacked

        result = self.controller.check_consensus(proposed_velocity)
        self.assertFalse(result)
        print(">> Consensus: DENIED (Attack Blocked)")


if __name__ == '__main__':
    unittest.main()
