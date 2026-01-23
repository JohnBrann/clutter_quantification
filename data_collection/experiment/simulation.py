from pathlib import Path
import time

import numpy as np
import pybullet

from utils.grasp import Label
from utils.perception import *
from experiment import btsim, workspace_lines
from utils.transform import Rotation, Transform
from utils.noise import apply_noise
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Slerp
import json


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, add_noise=False, sideview=False, save_dir=None, save_freq=8, remove_box=True, replica_scene_id=0):
        assert scene in ["pile", "packed", "real_packed","real_pile", "replica"]

        self.urdf_root = Path("object_sets")
        self.scene = scene
        self.object_set = object_set
        self.replica_scene_id = replica_scene_id
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq)
        self.gripper = Gripper(self.world)
        self.size = 6 * self.gripper.finger_depth
        # intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.intrinsic = CameraIntrinsic(848, 480, 426.678, 426.67822265625, 427.2525634765625, 234.44296264648438)
        self.camera = self.world.add_camera(self.intrinsic, 0.1, 2.0)

        self.remove_box = remove_box

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    # def discover_objects(self):
    #     root = self.urdf_root / self.object_set
    #     self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

     # Discovers objects in the structure expected from MOAD datasets
    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = sorted(
            (p for p in root.glob("*/fused/*.urdf") if p.is_file()),
            key=lambda p: p.stem
        )
        print(f'{self.object_urdfs}')

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        elif self.scene == "replica":
            replica_scene_path = Path("scene_replica_scenes") / "scenes"
            self.generate_replica_scene(table_height, self.replica_scene_id, replica_scene_path)
        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        nlz = 0.005
        self.lower = np.r_[lx, ly, lz]
        self.newlower = np.r_[lx, ly, nlz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            # print(f'{urdf} xy: {xy}')
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(1.0, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=3.0)

        # remove box
        if self.remove_box == True:
            self.world.remove_body(box)
            self.remove_and_wait()

        self.wait_for_objects_to_rest(timeout=3.0)
        
        # save pos and quaternion of each object to a json file
        self.save_poses_to_json()


    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

        self.wait_for_objects_to_rest(timeout=3.0)
        
        # positions = [
        #     {'x': 0.3, 'y': 0, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0, 'y': 0, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0.3, 'y': 0.3, 'angle': 0.0, 'scale': 1.0},
        #     {'x': 0, 'y': 0.3, 'angle': 0.0, 'scale': 1.0},
        # ]

        # # positions = [
        # #     {'x': 0, 'y': 0, 'angle': 0.0, 'scale': 1.0},
        # # ]
        # test_urdf = self.object_urdfs[5]

        # for pos in positions:
        #     urdf = test_urdf
        #     x, y = pos['x'], pos['y']
        #     angle = pos.get('angle', 0.0)
        #     scale = pos.get('scale', 1.0)
            
        #     z = 1.0
        #     # rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
        #     rotation = Rotation.random(random_state=self.rng)
        #     pose = Transform(rotation, np.r_[x, y, z])
        #     body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            
        #     lower, upper = self.world.p.getAABB(body.uid)
        #     z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
        #     body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
        #     self.world.step()
        #     self.wait_for_objects_to_rest(timeout=3.0)

        # save pos and quaternion of each object to a json file
        self.save_poses_to_json()


    def generate_replica_scene(self, table_height, scene_number, replica_scene_path=1, scale=1.0):
        # Get scene data
        full_scene_path = Path(replica_scene_path) / f"{scene_number}.npz"
        data = np.load(full_scene_path, allow_pickle=True)
        print(f'data: {data}')

        model_names_all = data["model_names"]
        poses_all = data["poses"]  # [x, y, z, qx, qy, qz, q]

        # Index into the requested scene (works for either multi-scene or single-scene)
        model_names = model_names_all[scene_number] if model_names_all.ndim == 2 else model_names_all
        poses = poses_all[scene_number] if poses_all.ndim == 3 else poses_all
        
        # extracts the end of each model name path so we can use it to match the obejct names in the .npz file
        urdf_by_name = {p.stem: p for p in self.object_urdfs}
        
        # place objects
        # spawned = []
        for model_name, pose7 in zip(model_names, poses):
            # Map model name -> urdf path
            urdf = urdf_by_name[model_name]
            
            x_, y_, z, qx, qy, qz, qw = pose7
            x = x_ + 0.15
            y = y_ + 0.15
            rot = Rotation.from_quat([qx, qy, qz, qw])  # scipy expects [x, y, z, w]

            # Load at recorded pose
            pose = Transform(rot, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)

            # Adjust z so it sits on the table (keep your existing behavior)
            lower, upper = self.world.p.getAABB(body.uid)
            z_on_table = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rot, np.r_[x, y, z_on_table]))

            self.world.step()
            self.wait_for_objects_to_rest(timeout=3.0)
            # spawned.append(body)

        self.wait_for_objects_to_rest(timeout=3.0)
        self.save_poses_to_json()
        # return spawned


    
    def save_poses_to_json(self, path="scene_poses.json"):
        """
        Save current world object poses to JSON.
        Each item contains: uid (int), name (str), pos [x,y,z], orn [x,y,z,w], optional scale.
        """
        out = {
            "objects": []
        }

        for uid, body in list(self.world.bodies.items()):
            try:
                uid_i = int(uid)
                pos, orn = self.world.p.getBasePositionAndOrientation(uid_i)
                name = getattr(body, "name", "") or ""
                # include any known scale if your bodies carry that info, else default 1.0
                scale = getattr(body, "scale", None)
                item = {
                    "uid": uid_i,
                    "name": name,
                    "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    # PyBullet ordering: (x, y, z, w)
                    "quat_xyzw": [float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])]
                }
                if scale is not None:
                    item["scale"] = float(scale)
                out["objects"].append(item)

                print(f"[info] saved uid={uid_i} name='{name}' pos={item['pos']} quat={item['quat_xyzw']}")
            except Exception as e:
                print(f"[warn] failed to read pose for uid={uid}: {e}")

        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[info] scene saved to {path}")




    def recovered_scene(self, mesh_list):
        # texture_id = world.p.loadTexture('/home/pinhao/Desktop/GIGA/texture_0.jpg')
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)
        for (mesh_path, scale, pose) in mesh_list:
            pose = Transform.from_matrix(pose)
            mesh_path = '_'.join(mesh_path.split('_')[:-1])+'.urdf'
            body = self.world.load_urdf(mesh_path, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            # body.set_pose(pose=pose)
            self.world.step()


    def execute_grasp(self, grasp, remove=True, allow_contact=True):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.2])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.2])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        # self.gripper.reset(T_world_pregrasp, opening_width=grasp.width)
        self.gripper.reset(T_world_pregrasp)
        #time.sleep(1)
        #print('calculate pregrasp',T_world_pregrasp.translation)
        #print('world pregrasp',self.gripper.body.get_pose().translation)
        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width, 'pregrasp'
            return result
            #print('pregrasp contact')
            #time.sleep(3)
        else:
            #print('non contact')
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=False)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width, 'grasp'
                quick_act = False
                if quick_act:
                    self.gripper.move(0.0)
                    self.advance_sim(10)
                    # need some time to check grasp or not, if this time is too short, failure of grasp is considered as drop
                    self.advance_sim(30)
                    if self.check_success(self.gripper):
                        dis_from_hand = self.gripper.get_distance_from_hand()
                        self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                        self.gripper.move_gripper_top_down()
                        shake_label = False
                        if self.check_success(self.gripper):
                            shake_label = self.gripper.shake_hand(dis_from_hand)
                            # print('finish shaking')
                        if self.check_success(self.gripper) and shake_label:
                            result = Label.SUCCESS, self.gripper.read(), 'success'
                        else:
                            result = Label.FAILURE, self.gripper.max_opening_width, 'after'
                        if remove:
                            contacts = self.world.get_contacts(self.gripper.body)
                            self.world.remove_body(contacts[0].bodyB)
                    else:
                        result = Label.FAILURE, self.gripper.max_opening_width, 'grasp'
            else:
                self.gripper.move(0.0)
                self.advance_sim(10)
                if self.check_success(self.gripper):
                    dis_from_hand = self.gripper.get_distance_from_hand()
                    self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                    self.gripper.move_gripper_top_down()
                    shake_label = False
                    if self.check_success(self.gripper):
                        shake_label = self.gripper.shake_hand(dis_from_hand)
                        #print('finish shaking')
                    if self.check_success(self.gripper) and shake_label:
                        result = Label.SUCCESS, self.gripper.read(),'success'
                        if remove:
                            contacts = self.world.get_contacts(self.gripper.body)
                            self.world.remove_body(contacts[0].bodyB)
                    else:
                        result =  Label.FAILURE, self.gripper.max_opening_width, 'after'
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width, 'after'
        self.world.remove_body(self.gripper.body)
        if remove:
            self.remove_and_wait()
        return result[:2]
    
    
    def advance_sim(self,frames):
        for _ in range(frames):
            self.world.step()

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res
    

    def idle(self, hz=240):
        """Keep the sim running without executing any grasps."""
        import time
        print("Scene spawned. Press Ctrl+C to exit.")
        try:
            while True:
                self.world.step()          # uses btsim.BtWorld.step()
                time.sleep(1.0 / hz)       # gentle throttle
        except KeyboardInterrupt:
            pass


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("object_sets/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )
    def grasp_object_id(self):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            # contact = contacts[0]
            # get rid body
            grased_id = contact.bodyB
            if grased_id.uid!=self.body.uid:
                return grased_id.uid
            
    def get_distance_from_hand(self,):
        object_id = self.grasp_object_id()
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        return dist_from_hand
    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
    
    def move_gripper_top_down(self):
        current_pose = self.body.get_pose()
        pos = current_pose.translation + 0.1
        flip = Rotation.from_euler('y', np.pi)
        target_ori = Rotation.identity()*flip
        self.move_tcp_pose(Transform(rotation=target_ori,translation=pos),abs=True)
    
    def move_tcp_pose(self, target, eef_step1=0.002, vel1=0.10, abs=False):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        pos_diff = target.translation - T_world_tcp.translation
        n_steps = max(int(np.linalg.norm(pos_diff) / eef_step1),10)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel1
        key_rots = np.stack((T_world_body.rotation.as_quat(),target.rotation.as_quat()),axis=0)
        key_rots = Rotation.from_quat(key_rots)
        slerp = Slerp([0.0,1.0],key_rots)
        times = np.linspace(0,1,n_steps)
        orientations = slerp(times).as_quat()
        for ii in range(n_steps):
            T_world_tcp.translation += dist_step
            T_world_tcp.rotation = Rotation.from_quat(orientations[ii])
            if abs is True:
                # todo by haojie add the relation transformation later
                self.constraint.change(
                    jointChildPivot=T_world_tcp.translation,
                    jointChildFrameOrientation=T_world_tcp.rotation.as_quat(),
                    maxForce=300,
                )
            else:
                self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
    
    def shake_hand(self,pre_dist):
        grasp_id = self.grasp_object_id()
        current_pose = self.body.get_pose()
        x,y,z = current_pose.translation[0],current_pose.translation[1],current_pose.translation[2]
        default_position = [x, y, z]
        shake_position = [x, y, z+0.05]
        hand_orientation2 = pybullet.getQuaternionFromEuler([np.pi, 0, -np.pi/2])
        shake_orientation1 = pybullet.getQuaternionFromEuler([np.pi, -np.pi / 12, -np.pi/2])
        shake_orientation2 = pybullet.getQuaternionFromEuler([np.pi, np.pi / 12, -np.pi/2])
        new_trans = current_pose.translation + np.array([0.,0.,0.05])
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2),translation=new_trans))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=shake_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation1), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        else:
            return True
        
    def is_dropped(self,object_id,prev_dist):
        pos,_ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        if np.isclose(prev_dist,dist_from_hand,atol=0.1):
            return False
        else:
            return True