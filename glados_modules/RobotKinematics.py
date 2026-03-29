"""Forward and inverse kinematics for the GLaDOS 4-DOF robot arm.

Computes the head camera's pointing direction from servo angles (FK),
and computes servo angles to achieve a desired pointing direction (IK).

The kinematic chain is: base -> body_LR -> body_UD -> head_LR -> head_UD -> camera.
All joint axes are taken from the URDF and are non-orthogonal. The current
tracking code treats LR and UD as independent axes, which causes cross-axis
oscillation ("nodding attack"). This module replaces that with proper 3D math.

Usage:
    kin = RobotKinematics(servo_middles, servo_mins, servo_maxs)
    yaw, pitch = kin.forward_kinematics(current_angles)
    head_targets = kin.inverse_kinematics_head(target_yaw, target_pitch, body_angles)
    body_targets = kin.inverse_kinematics_body(target_yaw, target_pitch)
"""

# builtin
from typing import Dict, Tuple
from math import radians, degrees, atan2, asin, cos, sin

# 3rd party
import numpy as np

# glados imports
from glados_modules.GladosEnums import KinematicsEnums, ServoEnum


class RobotKinematics:
    """Forward and inverse kinematics for the GLaDOS 4-DOF robot arm.

    Converts between servo angles (degrees) and the head camera's
    world-space pointing direction (yaw, pitch). Uses Rodrigues' rotation
    formula for FK and numerical Jacobian Newton-Raphson iteration for IK.

    The two-stage IK preserves head-locks-first behavior:
    - inverse_kinematics_head: solve head joints given fixed body position
    - inverse_kinematics_body: solve body joints with head at middle
    """

    def __init__(self, servo_middles: Dict[str, float],
                 servo_mins: Dict[str, float],
                 servo_maxs: Dict[str, float]) -> None:
        """Initialize with servo calibration data.

        Args:
            servo_middles: Dict mapping servo name -> center angle in degrees.
            servo_mins: Dict mapping servo name -> minimum angle in degrees.
            servo_maxs: Dict mapping servo name -> maximum angle in degrees.
        """
        self._middles = servo_middles
        self._mins = servo_mins
        self._maxs = servo_maxs

        # Joint ordering
        self._joint_order = list(KinematicsEnums.JOINT_ORDER.value)

        # Normalize and store joint axes as numpy arrays
        axes_raw = [
            KinematicsEnums.AXIS_BODY_LR.value,
            KinematicsEnums.AXIS_BODY_UD.value,
            KinematicsEnums.AXIS_HEAD_LR.value,
            KinematicsEnums.AXIS_HEAD_UD.value,
        ]
        self._axes = []
        for ax in axes_raw:
            a = np.array(ax, dtype=np.float64)
            a = a / np.linalg.norm(a)
            self._axes.append(a)

        # Per-joint sign multipliers
        self._signs = [
            KinematicsEnums.SIGN_BODY_LR.value,
            KinematicsEnums.SIGN_BODY_UD.value,
            KinematicsEnums.SIGN_HEAD_LR.value,
            KinematicsEnums.SIGN_HEAD_UD.value,
        ]

        # Camera optical axis in the last link frame
        cam_ax = np.array(KinematicsEnums.CAMERA_OPTICAL_AXIS_LOCAL.value, dtype=np.float64)
        self._camera_axis = cam_ax / np.linalg.norm(cam_ax)

        # IK solver parameters
        self._ik_max_iter = KinematicsEnums.IK_MAX_ITERATIONS.value
        self._ik_tol = KinematicsEnums.IK_TOLERANCE_DEG.value
        self._ik_damping = KinematicsEnums.IK_DAMPING_LAMBDA.value
        self._ik_eps = KinematicsEnums.IK_FINITE_DIFF_EPS.value

    def _servo_to_rad(self, joint_index: int, servo_deg: float) -> float:
        """Convert servo degrees to radians from home position.

        Args:
            joint_index: Index into the joint order (0-3).
            servo_deg: Servo angle in degrees.

        Returns:
            Rotation angle in radians, sign-corrected for URDF convention.
        """
        name = self._joint_order[joint_index]
        return self._signs[joint_index] * radians(servo_deg - self._middles[name])

    def _rad_to_servo(self, joint_index: int, angle_rad: float) -> float:
        """Convert radians from home position to servo degrees, clamped.

        Args:
            joint_index: Index into the joint order (0-3).
            angle_rad: Rotation angle in radians.

        Returns:
            Servo angle in degrees, clamped to physical range.
        """
        name = self._joint_order[joint_index]
        servo_deg = degrees(angle_rad) / self._signs[joint_index] + self._middles[name]
        return max(self._mins[name], min(self._maxs[name], servo_deg))

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rodrigues' rotation formula: rotation matrix for angle about arbitrary axis.

        Args:
            axis: 3D unit vector defining the rotation axis.
            angle_rad: Rotation angle in radians.

        Returns:
            3x3 rotation matrix.
        """
        c = cos(angle_rad)
        s = sin(angle_rad)
        t = 1.0 - c
        x, y, z = axis
        return np.array([
            [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ], dtype=np.float64)

    def _fk_rotation(self, joint_angles_rad: list) -> np.ndarray:
        """Compute cumulative rotation matrix through the kinematic chain.

        Args:
            joint_angles_rad: List of 4 joint angles in radians (from home).

        Returns:
            3x3 rotation matrix from base frame to camera frame.
        """
        r_cumulative = np.eye(3, dtype=np.float64)
        for i in range(4):
            # Transform the joint axis to the current world orientation
            axis_world = r_cumulative @ self._axes[i]
            r_joint = self._rotation_matrix(axis_world, joint_angles_rad[i])
            r_cumulative = r_joint @ r_cumulative
        return r_cumulative

    def forward_kinematics_vector(self, servo_angles: Dict[str, float]) -> np.ndarray:
        """Compute the camera's pointing direction as a 3D unit vector.

        Args:
            servo_angles: Dict mapping servo name -> current angle in degrees.

        Returns:
            3D unit vector in world frame representing camera optical axis.
        """
        angles_rad = [self._servo_to_rad(i, servo_angles[self._joint_order[i]])
                      for i in range(4)]
        r_total = self._fk_rotation(angles_rad)
        return r_total @ self._camera_axis

    def forward_kinematics(self, servo_angles: Dict[str, float]) -> Tuple[float, float]:
        """Compute world-space yaw and pitch from current servo angles.

        Args:
            servo_angles: Dict mapping servo name -> current angle in degrees.

        Returns:
            Tuple of (yaw_degrees, pitch_degrees). Yaw=0 is forward (along Z),
            positive yaw = left (positive X). Pitch=0 is horizontal,
            positive pitch = up (positive Y).
        """
        direction = self.forward_kinematics_vector(servo_angles)
        return self._pointing_to_yaw_pitch(direction)

    @staticmethod
    def _pointing_to_yaw_pitch(direction: np.ndarray) -> Tuple[float, float]:
        """Convert a 3D unit vector to (yaw, pitch) in degrees.

        Args:
            direction: 3D unit vector.

        Returns:
            Tuple of (yaw_degrees, pitch_degrees).
        """
        # Yaw: rotation about Y axis, measured from Z axis toward X axis
        yaw = degrees(atan2(direction[0], direction[2]))
        # Pitch: elevation above the XZ plane
        pitch = degrees(asin(np.clip(direction[1], -1.0, 1.0)))
        return yaw, pitch

    @staticmethod
    def _yaw_pitch_to_pointing(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """Convert (yaw, pitch) in degrees to a 3D unit vector.

        Args:
            yaw_deg: Yaw angle in degrees.
            pitch_deg: Pitch angle in degrees.

        Returns:
            3D unit vector.
        """
        yaw_r = radians(yaw_deg)
        pitch_r = radians(pitch_deg)
        return np.array([
            cos(pitch_r) * sin(yaw_r),
            sin(pitch_r),
            cos(pitch_r) * cos(yaw_r),
        ], dtype=np.float64)

    def _compute_jacobian_2x2(self, servo_angles: Dict[str, float],
                               variable_indices: list) -> np.ndarray:
        """Compute 2x2 Jacobian of (yaw, pitch) w.r.t. two variable joints.

        Uses numerical finite differences on the FK.

        Args:
            servo_angles: Current servo angles in degrees.
            variable_indices: List of 2 joint indices to differentiate w.r.t.

        Returns:
            2x2 numpy array where J[i,j] = d(output_i)/d(servo_j) in deg/deg.
        """
        eps_deg = degrees(self._ik_eps)
        yaw0, pitch0 = self.forward_kinematics(servo_angles)
        j = np.zeros((2, 2), dtype=np.float64)
        for col, joint_idx in enumerate(variable_indices):
            name = self._joint_order[joint_idx]
            perturbed = dict(servo_angles)
            perturbed[name] = servo_angles[name] + eps_deg
            yaw1, pitch1 = self.forward_kinematics(perturbed)
            j[0, col] = (yaw1 - yaw0) / eps_deg
            j[1, col] = (pitch1 - pitch0) / eps_deg
        return j

    def _ik_solve_2x2(self, target_yaw: float, target_pitch: float,
                       servo_angles: Dict[str, float],
                       variable_indices: list) -> Dict[str, float]:
        """Solve IK for 2 joints using damped Jacobian Newton-Raphson.

        Args:
            target_yaw: Desired yaw in degrees.
            target_pitch: Desired pitch in degrees.
            servo_angles: Starting servo angles (all 4 joints) in degrees.
            variable_indices: Which 2 joint indices to solve for.

        Returns:
            Dict with updated servo angles for the variable joints, clamped.
        """
        angles = dict(servo_angles)

        for _ in range(self._ik_max_iter):
            yaw_cur, pitch_cur = self.forward_kinematics(angles)
            err_yaw = target_yaw - yaw_cur
            err_pitch = target_pitch - pitch_cur

            if abs(err_yaw) < self._ik_tol and abs(err_pitch) < self._ik_tol:
                break

            error = np.array([err_yaw, err_pitch], dtype=np.float64)
            j = self._compute_jacobian_2x2(angles, variable_indices)

            # Damped least-squares: dq = J^T (J J^T + lambda^2 I)^-1 error
            jjt = j @ j.T
            jjt_damped = jjt + (self._ik_damping ** 2) * np.eye(2)
            dq = j.T @ np.linalg.solve(jjt_damped, error)

            # Apply and clamp
            for col, joint_idx in enumerate(variable_indices):
                name = self._joint_order[joint_idx]
                new_val = angles[name] + dq[col]
                angles[name] = max(self._mins[name], min(self._maxs[name], new_val))

        result = {}
        for joint_idx in variable_indices:
            name = self._joint_order[joint_idx]
            result[name] = angles[name]
        return result

    def inverse_kinematics_head(self, target_yaw: float, target_pitch: float,
                                 body_angles: Dict[str, float]) -> Dict[str, float]:
        """Compute head servo angles to point at target, given fixed body angles.

        This is the "head locks first" half of the IK. Given the current body
        servo positions, find head_LR and head_UD angles that point the camera
        at the desired world-space direction.

        Args:
            target_yaw: Desired world-space yaw in degrees.
            target_pitch: Desired world-space pitch in degrees.
            body_angles: Dict with body_left_right and body_up_down angles in degrees.

        Returns:
            Dict with head_left_right and head_up_down target angles in degrees.
        """
        head_lr_name = self._joint_order[2]
        head_ud_name = self._joint_order[3]

        # Start head joints at their current middle (home) as initial guess
        starting_angles = dict(body_angles)
        starting_angles[head_lr_name] = self._middles[head_lr_name]
        starting_angles[head_ud_name] = self._middles[head_ud_name]

        return self._ik_solve_2x2(target_yaw, target_pitch,
                                   starting_angles, [2, 3])

    def inverse_kinematics_body(self, target_yaw: float,
                                 target_pitch: float) -> Dict[str, float]:
        """Compute ideal body servo angles for a target direction.

        This is the "body follows" half. Computes where the body should
        eventually end up so the head can be near its home position.
        Solves with head joints fixed at their middle positions.

        Args:
            target_yaw: Desired world-space yaw in degrees.
            target_pitch: Desired world-space pitch in degrees.

        Returns:
            Dict with body_left_right and body_up_down target angles in degrees.
        """
        body_lr_name = self._joint_order[0]
        body_ud_name = self._joint_order[1]
        head_lr_name = self._joint_order[2]
        head_ud_name = self._joint_order[3]

        # Start body at middle, head at middle
        starting_angles = {
            body_lr_name: self._middles[body_lr_name],
            body_ud_name: self._middles[body_ud_name],
            head_lr_name: self._middles[head_lr_name],
            head_ud_name: self._middles[head_ud_name],
        }

        return self._ik_solve_2x2(target_yaw, target_pitch,
                                   starting_angles, [0, 1])
