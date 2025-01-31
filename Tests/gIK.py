#!/usr/bin/env python3
"""
Demo script to load a robot from a URDF file using IKPy and display it
on a 3D Matplotlib grid so that movement can be visualized.
"""

import numpy as np
import matplotlib.pyplot as plt
from ikpy.chain import Chain
from typing import List


def plot_robot_configuration(
    chain: Chain,
    joint_angles: List[float],
    ax: plt.Axes,
    show_grid: bool = True
) -> None:
    """
    Plot the robot's configuration given the chain and joint angles on a 3D Matplotlib axis.

    :param chain: An IKPy Chain object representing the robot.
    :param joint_angles: A list of joint angle values in radians.
    :param ax: A Matplotlib Axes object (in 3D) on which to plot the robot.
    :param show_grid: If True, display the grid on the 3D plot.
    """
    # Clear the axis to allow re-plotting if called repeatedly.
    ax.cla()

    # Plot the robot at the given joint angles using IKPy's built-in plotting function.
    chain.plot(joint_angles, ax=ax)

    # Optional: Customize the 3D axis limits for better visualization.
    # (Change according to your robot's scale)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.0)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    if show_grid:
        ax.grid(True)


def main() -> None:
    """
    Main function to demonstrate how to load a URDF file into an IKPy Chain
    and visualize robot movements on a 3D Matplotlib plot.
    """
    # --------------------------------------------------------------------------
    # 1. LOAD THE ROBOT CHAIN FROM URDF
    # --------------------------------------------------------------------------
    urdf_path = "GLaDOS.urdf"  # Replace with your URDF file path
    robot_chain: Chain = Chain.from_urdf_file(urdf_path, base_elements=["ceiling_link"])
    robot_chain.active_links_mask = [False, True, True, True, True]
    # --------------------------------------------------------------------------
    # 2. SETUP MATPLOTLIB FIGURE & AXES
    # --------------------------------------------------------------------------
    fig = plt.figure("Robot Visualization")
    ax = fig.add_subplot(111, projection='3d')

    # --------------------------------------------------------------------------
    # 3. DEFINE/INITIALIZE JOINT ANGLES
    #    This list must match the number of actuated joints in your URDF.
    # --------------------------------------------------------------------------
    # For demonstration, we'll just set each joint to 0 rad initially.
    # If your robot has, for example, 6 actuated joints, use [0.0]*6
    # Adjust the length of this list to match your robot.
    current_joint_angles = [0.0] * 5

    # --------------------------------------------------------------------------
    # 4. PLOT THE INITIAL CONFIGURATION OF THE ROBOT
    # --------------------------------------------------------------------------
    plot_robot_configuration(robot_chain, current_joint_angles, ax)
    plt.pause(0.5)  # Pause briefly to show the initial configuration

    # --------------------------------------------------------------------------
    # 5. DEMONSTRATE MOVEMENT OF THE ROBOT
    #    We'll do a simple loop that modifies each joint in a small range.
    # --------------------------------------------------------------------------
    num_steps = 40
    joint_range = np.linspace(0.0, np.pi / 6, num_steps)  # Joint moves from 0 to pi/6
    num_joints = 4
    for step in range(num_steps):
        # Update all joints to the same angle for demonstration
        for j in range(num_joints):
            current_joint_angles[j] = joint_range[step]

        # Clear and re-plot the robot in the new configuration
        plot_robot_configuration(robot_chain, current_joint_angles, ax)

        # Render the updated plot
        plt.draw()
        plt.pause(0.1)  # Pause to create an animation-like effect

    # Keep the final plot open
    plt.show()


if __name__ == "__main__":
    main()
