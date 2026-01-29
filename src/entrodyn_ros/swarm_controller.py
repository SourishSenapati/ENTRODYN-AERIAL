try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    # Windows/Testing Fallback
    # This allows the code to be imported and tested without a full ROS2 installation
    class Node:
        """Mock Node for Testing/Windows"""

        def __init__(self, name):
            self.node_name = str(name)

        def get_logger(self):
            return self

        def warn(self, msg):
            print(f"[LOG:WARN] {msg}")

        def info(self, msg):
            print(f"[LOG:INFO] {msg}")

import numpy as np


class SwarmController(Node):
    """
    Main Swarm Control Node for ENTRODYN-AERIAL.
    Manages formation, entropy-gradient tracking, and consensus.
    """

    def __init__(self):
        super().__init__('swarm_controller')
        self.peer_drones = []  # List of connected drone interfaces

    def check_consensus(self, proposed_velocity):
        """
        PBFT Logic: 
        Wait for 2/3rds of swarm to confirm vector before moving.
        Eliminates 'rogue' drone errors.
        """
        votes = 0
        # required_quorum = int(len(self.peer_drones) * 0.67) + 1
        # For demo purposes, let's assume a fixed quorum if peers aren't populated
        if not self.peer_drones:
            return True  # Bypass for solo testing

        required_quorum = int(len(self.peer_drones) * 0.67) + 1

        for peer in self.peer_drones:
            if peer.validate_vector(proposed_velocity):
                votes += 1

        if votes >= required_quorum:
            return True  # Execute Move
        else:
            self.get_logger().warn("Byzantine Fault Detected! Aborting rogue command.")
            return False
