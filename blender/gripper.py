#!/usr/bin/env blender --python

import bpy
import os
from math import pi

def create_empty(name):
    #obj = bpy.data.objects.new(name, None)
    #bpy.context.collection.objects.link(obj)
    bpy.ops.object.empty_add(type="PLAIN_AXES", radius=0.1)
    #obj.scale = (0.1, 0.1, 0.1)
    #bpy.ops.object.transform_apply()
    #return obj
    bpy.context.object.name = name
    return bpy.context.object

# def import_stl(filepath):
#     parent = create_empty(filepath)
#     bpy.ops.import_mesh.stl(filepath=filepath)
#     for i in bpy.context.selected_objects:
#         i.parent = parent
#     return parent

# def import_obj(filepath):
#     parent = create_empty(filepath)
#     bpy.ops.import_scene.obj(filepath=filepath)
#     for i in bpy.context.selected_objects:
#         if i.type == "MESH":
#             # Set each mesh's parent
#             i.parent = parent
#         else:
#             # delete lights, cameras, and any other non-mesh objects
#             bpy.data.objects.remove(i, do_unlink=True)
#     return parent

# def import_collada(filepath):
#     parent = create_empty(filepath)
#     bpy.ops.wm.collada_import(filepath=filepath)
#     for i in bpy.context.selected_objects:
#         if i.type == "MESH":
#             # Set each mesh's parent
#             i.parent = parent
#         else:
#             # delete lights, cameras, and any other non-mesh objects
#             bpy.data.objects.remove(i, do_unlink=True)
#     return parent


def import_mesh(name, filepath):
    parent = create_empty(name)

    _, ext = os.path.splitext(filepath)
    if ".dae" == ext.lower():
        bpy.ops.wm.collada_import(filepath=filepath)
    elif ".stl" == ext.lower():
        bpy.ops.import_mesh.stl(filepath=filepath)
    elif ".obj" == ext.lower():
        bpy.ops.import_scene.obj(filepath=filepath)
    else:
        return None

    for i in bpy.context.selected_objects:
        if i.type == "MESH":
            # Set each mesh's parent
            i.parent = parent
            i.name = name + '.' + i.name
        else:
            # delete lights, cameras, and any other non-mesh objects
            bpy.data.objects.remove(i, do_unlink=True)

    bpy.context.view_layer.objects.active = parent
    parent.select_set(True)
    return parent

class Gripper:
    def __init__(self):
        self._links = {}
        self._joints = {}
        self._mesh_path = "robotiq_2f_85_gripper_visualization/meshes/"
        self._grasp_point = create_empty("grasp_point")
        self._robotiq_arg2f_85()
        base = self._links["robotiq_arg2f_base_link"]
        base.parent = self._grasp_point
        base.location = (0, 0, -0.125)

    def _robotiq_arg2f_85(self):
        self._robotiq_arg2f_base_link()
        self._finger_links(fingerprefix="left", stroke=85)
        self._finger_links(fingerprefix="right", stroke=85)
        self._finger_joint()
        self._right_outer_knuckle_joint()
        self._robotiq_arg2f_transmission()

    def _link(self, name, visual, scale, color):
        mesh = import_mesh(name, self._mesh_path + visual)
        bpy.ops.transform.resize(value=(scale, scale, scale))
        bpy.ops.object.transform_apply()
        self._links[name] = mesh

    def _joint(self, name, type, parent, child, axis=(0.0, 0.0, 1.0), xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), q_min=0.0, q_max=0.0):
        origin = create_empty(name+"_origin")
        origin.parent = self._links[parent]
        origin.location = xyz
        origin.rotation_euler = rpy
        joint = create_empty(name)
        joint.parent = origin
        joint.rotation_mode = 'AXIS_ANGLE'
        joint.rotation_axis_angle = (0.0, axis[0], axis[1], axis[2])
        self._links[child].parent = joint
        self._joints[name] = { "type": type, "axis": axis, "q_min": q_min, "q_max": q_max, "joint": joint }

    def _robotiq_arg2f_base_link(self):
        self._link(
            name="robotiq_arg2f_base_link",
            visual="visual/robotiq_arg2f_85_base_link.dae",
            scale=0.001,
            color=(0.1, 0.1, 0.1, 1.0))

    def _finger_links(self, fingerprefix, stroke):
        self._outer_knuckle(fingerprefix=fingerprefix, stroke=stroke)
        self._outer_finger(fingerprefix=fingerprefix, stroke=stroke)
        self._inner_finger(fingerprefix=fingerprefix, stroke=stroke)
        self._inner_finger_pad(fingerprefix=fingerprefix)
        self._inner_knuckle(fingerprefix=fingerprefix)

    def _outer_knuckle(self, fingerprefix, stroke):
        self._link(
            name=fingerprefix+"_outer_knuckle",
            visual="visual/robotiq_arg2f_85_outer_knuckle.dae",
            scale=0.001,
            color=(0.792156862745098, 0.819607843137255, 0.933333333333333, 1.0))

    def _outer_finger(self, fingerprefix, stroke):
        self._link(
            name=fingerprefix+"_outer_finger",
            visual="visual/robotiq_arg2f_85_outer_finger.dae",
            scale=0.001,
            color=(0.1, 0.1, 0.1, 1.0))

    def _inner_finger(self, fingerprefix, stroke):
        self._link(
            name=fingerprefix+"_inner_finger",
            visual="visual/robotiq_arg2f_85_inner_finger.dae",
            scale=0.001,
            color=(0.1, 0.1, 0.1, 1.0))

    def _inner_finger_pad(self, fingerprefix):
        # TODO:           <box size="0.022 0.00635 0.0375"/>
        # TODO:           <color rgba="0.9 0.9 0.9 1" />
        name=fingerprefix+"_inner_finger_pad"
        bpy.ops.mesh.primitive_cube_add(size=1.0, enter_editmode=False)
        bpy.context.object.name = name
        self._links[name] = bpy.context.object
        bpy.ops.transform.resize(value=(0.022, 0.00635, 0.0375))
        bpy.ops.object.transform_apply()

    def _inner_knuckle(self, fingerprefix):
        self._link(
            name=fingerprefix+"_inner_knuckle",
            visual="visual/robotiq_arg2f_85_inner_knuckle.dae",
            scale=0.001,
            color=(0.1, 0.1, 0.1, 1.0))

    def _finger_joint(self):
        self._joint(
            name="finger_joint",
            type="revolute",
            parent="robotiq_arg2f_base_link",
            child="left_outer_knuckle",
            axis=(1.0, 0.0, 0.0),
            xyz=(0.0, -0.0306011, 0.054904),
            rpy=(0.0, 0.0, pi),
            q_min=0, q_max=0.8)
        self._finger_joints(fingerprefix="left", reflect=1.0)

    def _right_outer_knuckle_joint(self):
        self._joint(
            name="right_outer_knuckle_joint",
            type="revolute",
            parent="robotiq_arg2f_base_link",
            child="right_outer_knuckle",
            axis=(1.0, 0.0, 0.0),
            xyz=(0.0, 0.0306011, 0.054904),
            rpy=(0.0, 0.0, 0.0),
            q_min=0.0, q_max=0.81)
        self._finger_joints(fingerprefix="right", reflect=-1.0)

    def _finger_joints(self, fingerprefix, reflect):
        self._outer_finger_joint(fingerprefix)
        self._inner_knuckle_joint(fingerprefix, reflect)
        self._inner_finger_joint(fingerprefix)
        self._inner_finger_pad_joint(fingerprefix)

    def _outer_finger_joint(self, fingerprefix):
        self._joint(
            name=fingerprefix+"_outer_finger_joint",
            type="fixed",
            parent=fingerprefix+"_outer_knuckle",
            child=fingerprefix+"_outer_finger",
            xyz=(0.0, 0.0315, -0.0041))

    def _inner_knuckle_joint(self, fingerprefix, reflect):
        self._joint(
            name=fingerprefix+"_inner_knuckle_joint",
            type="revolute",
            xyz=(0.0, reflect * -0.0127, 0.06142),
            rpy=(0.0, 0.0, (1 + reflect) * pi / 2),
            parent="robotiq_arg2f_base_link",
            child=fingerprefix+"_inner_knuckle",
            axis=(1.0, 0.0, 0.0),
            q_min=0.0, q_max=0.8757)

    def _inner_finger_joint(self, fingerprefix):
        self._joint(
            name=fingerprefix+"_inner_finger_joint",
            type="revolute",
            xyz=(0.0, 0.0061, 0.0471),
            parent=fingerprefix+"_outer_finger",
            child=fingerprefix+"_inner_finger",
            axis=(1.0, 0.0, 0.0),
            q_min=0.0, q_max=0.8757)

    def _inner_finger_pad_joint(self, fingerprefix):
        self._joint(
            name=fingerprefix+"_inner_finger_pad_joint",
            type="fixed",
            xyz=(0.0, -0.0220203446692936, 0.03242),
            parent=fingerprefix+"_inner_finger",
            child=fingerprefix+"_inner_finger_pad")

    def _robotiq_arg2f_transmission(self):
        pass

    def set_config(self, q):
        self._joints["finger_joint"]["joint"].rotation_axis_angle[0] = q
        self._joints["right_inner_knuckle_joint"]["joint"].rotation_axis_angle[0] = q
        self._joints["left_inner_knuckle_joint"]["joint"].rotation_axis_angle[0] = q
        self._joints["right_inner_finger_joint"]["joint"].rotation_axis_angle[0] = -q
        self._joints["left_inner_finger_joint"]["joint"].rotation_axis_angle[0] = -q
        self._joints["right_outer_knuckle_joint"]["joint"].rotation_axis_angle[0] = q

    def set_material(self, mat):
        def recur(obj):
            if "MESH" == obj.type:
                obj.active_material = mat
                print(obj)
                
            for c in obj.children:
                recur(c)
                
        for link in self._links.values():
            recur(link)

def create_shadeless_material():
    #tree = bpy.context.scene.node_tree
    mat = bpy.data.materials.new(name="Flat")
    mat.use_nodes = True
    out = mat.node_tree.nodes[0]
    rgb = mat.node_tree.nodes.new(type="ShaderNodeRGB")
    rgb.outputs[0].default_value = (1, 0, 0, 1) # RGBA = red
    mat.node_tree.links.new(rgb.outputs[0], out.inputs[0])
    return mat


if "__main__" == __name__:
    # Delete the starting cube
    bpy.ops.object.delete(use_global=False)

    mat = create_shadeless_material()
    gripper = Gripper()
    gripper.set_material(mat)
    #gripper.set_config(0.5)

    z_near = 0.5
    z_far = 1.5


    # position the camera 1 meter above the view plane.
    # Objects at the origin will be centered in the rendered view
    camera = bpy.data.objects['Camera']
    camera.location = (0.0, 0.0, 1.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.clip_start = z_near
    camera.data.clip_end = z_far
