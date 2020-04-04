#!/usr/bin/env blender --python

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) UC Berkeley AUTOLAB
#
# This file loads a mesh and renders depth map views of it using
# Blender.  Tested on Belnder 2.82.  Most likely not compatible with
# 2.79 and earlier.
#
# Launching Option 1:
#   ./render_depth.py
#
# Launching Option 2:
#   /path/to/blender --python render_depth.py
#
# Authors:
#   Jeff Ichnowski (jeffi@berkeley.edu)

import bpy
import os
import sys
from random import Random
from mathutils import Quaternion
from math import pi, sin, cos, sqrt

# Hacky... force python to look for modules in this directory.
# Supposedly this is the default behavior, but blender must somehow be
# disabling it
sys.path.append(os.getcwd())

from gripper import Gripper

def create_empty(name):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    return obj

def import_stl(filepath):
    parent = create_empty(filepath)
    bpy.ops.import_mesh.stl(filepath=filepath)
    for i in bpy.context.selected_objects:
        i.parent = parent
        i.select_set(False)
    return parent

def import_obj(filepath):
    parent = create_empty(filepath)
    bpy.ops.import_scene.obj(filepath=filepath)
    for i in bpy.context.selected_objects:
        if i.type == "MESH":
            # Set each mesh's parent
            i.parent = parent
            i.select_set(False)
        else:
            # delete lights, cameras, and any other non-mesh objects
            bpy.data.objects.remove(i, do_unlink=True)
    return parent

def import_collada(filepath):
    parent = create_empty(filepath)
    bpy.ops.wm.collada_import(filepath=filepath)
    for i in bpy.context.selected_objects:
        if i.type == "MESH":
            # Set each mesh's parent
            i.parent = parent
            i.select_set(False)
        else:
            # delete lights, cameras, and any other non-mesh objects
            bpy.data.objects.remove(i, do_unlink=True)
    return parent


def import_mesh(filepath):
    _, ext = os.path.splitext(filepath)
    if ".dae" == ext.lower():
        mesh = import_collada(filepath)
    elif ".stl" == ext.lower():
        mesh = import_stl(filepath)
    elif ".obj" == ext.lower():
        mesh = import_obj(filepath)
    else:
        return None
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    return mesh

def random_quaternion(rng):
    # See: http://planning.cs.uiuc.edu/node198.html
    u1 = rng.random()
    u2 = rng.uniform(0.0, 2.0*pi)
    u3 = rng.uniform(0.0, 2.0*pi)

    r1 = sqrt(u1)
    i1 = sqrt(1 - u1)
    s2 = sin(u2)
    c2 = cos(u2)
    s3 = sin(u3)
    c3 = cos(u3)
    q = (i1 * s2, i1 * c2, r1 * s2, r1 * c2)
    return Quaternion(q)

def random_quaternion_uniform30(rng):
    "from https://www.mathworks.com/help/robotics/ref/quaternion.rotvec.html"
    angle = rng.uniform(0,pi/6)
    while True:
        x = rng.gauss(0,1)
        y = rng.gauss(0,1)
        z = rng.gauss(0,1)
        if x*x + y*y + z*z <= 1:
            break
    norm = (x*x + y*y + z*z) ** 0.5
    x,y,z = x/norm, y/norm, z/norm
    # qr = cos(angle/2)
    # s = sin(angle/2)
    # qi,qj,qk = x*s, y*s, z*s
    # q = (qi,qj,qk,qr)
    return Quaternion((x,y,z),angle)

def setup_depth_rendering(z_near, z_far):
    bpy.context.scene.use_nodes = True
    bpy.context.scene.world.use_nodes = True
    tree = bpy.context.scene.node_tree

    # We set up depth rendering using compositor nodes.
    # The mapping goes from:
    #   render.z -> map_range -> composite.image
    # Where render.z is the z-value of what is rendered,
    # map_range scales the values from 0 to 1, and
    # composit.image is what is rendered.

    map_range = tree.nodes.new(type="CompositorNodeMapRange")
    # Clamp to the output range.  This is probably not necessary but
    # can't hurt
    map_range.use_clamp = True
    map_range.inputs[1].default_value = z_near # minimum rendered 'Z' value
    map_range.inputs[2].default_value = z_far # maximum rendered 'Z' value
    # inputs[3] is the minimum output 'Z' value, which defaults to 0, which is what we want
    # inputs[4] is the maximum output 'Z' value, which defaults to 1, also what we want

    # these two nodes exist by default.  We could delete them and recreate.
    render_layers = tree.nodes["Render Layers"]
    composite = tree.nodes["Composite"]

    # link render_layer's 'Z' to map_range's input, and the composite's Z value
    tree.links.new(render_layers.outputs[2], map_range.inputs[0])
    tree.links.new(render_layers.outputs[2], composite.inputs[2])
    # link map_range's output to composite's input
    tree.links.new(map_range.outputs[0], composite.inputs[0])

def create_shadeless_material(rgba=(1,0,0,1)):
    #tree = bpy.context.scene.node_tree
    mat = bpy.data.materials.new(name="Flat")
    mat.use_nodes = True
    out = mat.node_tree.nodes[0]
    rgb = mat.node_tree.nodes.new(type="ShaderNodeRGB")
    rgb.outputs[0].default_value = rgba
    mat.node_tree.links.new(rgb.outputs[0], out.inputs[0])
    return mat

def setup_dual_rendering(z_near, z_far, output_directory, depth_path, rgb_path):
    # This method sets up two output files, one for depth, and one for
    # rgb.  After running this method, the compositor node contains
    # the output file, not the scene panel.
    #
    # For more explanation, see the answer on:
    # https://blender.stackexchange.com/questions/74086/how-do-i-batch-render-multiple-scenes-with-animation-settings
    bpy.context.scene.use_nodes = True
    bpy.context.scene.world.use_nodes = True
    tree = bpy.context.scene.node_tree

    # First remove all nodes from the scene
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create the output node
    file_node = tree.nodes.new(type='CompositorNodeOutputFile')
    file_node.base_path = output_directory
    file_node.format.file_format = 'OPEN_EXR'
    file_node.format.color_mode = 'RGBA'

    #for s in file_node.file_slots:
    #    file_node.file_slots.remove(file_node.inputs[0])
    file_node.file_slots.clear()

    render_layers = tree.nodes.new(type='CompositorNodeRLayers')

    # Set the background color
    # This is slightly roundabout, but since we turned on Nodes we set it in the nodes.
    bpy.data.worlds[0].node_tree.nodes['Background'].inputs['Color'].default_value = (0,0,0,1)

    # We may make these options later
    render_rgb = True
    render_depth = True

    if render_rgb:
        fin = file_node.file_slots.new('rgb')
        fs = file_node.file_slots['rgb']
        fs.use_node_format = True
        fs.path = rgb_path
        tree.links.new(render_layers.outputs['Image'], fin)

    if render_depth:
        fin = file_node.file_slots.new('depth')
        fs = file_node.file_slots['depth']
        fs.use_node_format = False
        fs.path = depth_path
        fs.format.file_format = 'OPEN_EXR'
        fs.format.color_depth = '32'
        fs.format.color_mode = 'RGB'

        # We set up depth rendering using compositor nodes.
        # The mapping goes from:
        #   render.z -> map_range -> composite.image
        # Where render.z is the z-value of what is rendered,
        # map_range scales the values from 0 to 1, and
        # composit.image is what is rendered.

        map_range = tree.nodes.new(type="CompositorNodeMapRange")
        # Clamp to the output range.  This is probably not necessary but
        # can't hurt
        map_range.use_clamp = True
        map_range.inputs[1].default_value = z_near # minimum rendered 'Z' value
        map_range.inputs[2].default_value = z_far # maximum rendered 'Z' value
        # inputs[3] is the minimum output 'Z' value, which defaults to 0, which is what we want
        # inputs[4] is the maximum output 'Z' value, which defaults to 1, also what we want

        # these two nodes exist by default.  We could delete them and recreate.

        # link render_layer's 'Z' to map_range's input, and the composite's Z value
        tree.links.new(render_layers.outputs['Depth'], map_range.inputs[0])
        # link map_range's output to composite's input
        tree.links.new(map_range.outputs[0], fin)

def set_material_recursive(obj, mat):
    if "MESH" == obj.type:
        obj.active_material = mat

    for c in obj.children:
        set_material_recursive(c, mat)

if "__main__" == __name__:
    # Delete the starting cube
    bpy.ops.object.delete(use_global=False)

    # ======================================== Options
    # Set these values to specify the depth range of the camera
    # Want to make sure the values are close to the actual depth range values.
    z_near = 0.6
    z_far = 0.85
    # Number of frames to generate
    num_pairs = 10
    # Input mesh
    mesh_file = 'nozzle_5222583.stl'
    # mesh_file = 'banana_3dnet/banana_3dnet.obj'
    #mesh_file = 'elephant_1_3dnet/elephant_1_3dnet.obj'
    # Output prefix
    # quaternion_output_file = os.path.join(os.getcwd(), "quaternion_hex_vase.txt")
    # depth_output_prefix = os.path.join(os.getcwd(), "depth_hex_vase_")
    # quaternion_output_file = os.path.join(os.getcwd(), "banana_3dnet/banana_3dnet.txt")
    # depth_output_prefix = os.path.join(os.getcwd(), "banana_3dnet/banana_3dnet_")
    # quaternion_output_file = os.path.join(os.getcwd(), "elephant_1_3dnet/elephant_1_3dnet.txt")
    # depth_output_prefix = os.path.join(os.getcwd(), "elephant_1_3dnet/elephant_1_3dnet_depth_")
    # rgb_output_prefix = os.path.join(os.getcwd(), "elephant_1_3dnet/elephant_1_3dnet_rgb_")

    output_path = os.path.join(os.getcwd(), "render_output")

    mesh_base, _ = os.path.splitext(mesh_file)
    quaternion_output_file = mesh_base + "_quat.txt"
    depth_output_prefix = mesh_base + "_depth_"
    rgb_output_prefix = mesh_base + "_rgb_"

    # ======================================== End of Options

    # position the camera 1 meter above the view plane.
    # Objects at the origin will be centered in the rendered view
    camera = bpy.data.objects['Camera']
    camera.location = (0.0, 0.0, 0.8)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.clip_start = z_near
    camera.data.clip_end = z_far
    # camera.data.angle = 27.0*pi/180.0
    camera.data.angle = 9.0*pi/180.0
    # Adding a table to the scene
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    table = bpy.context.active_object
    table.location = (0, 0, 0.03)

    # setup_depth_rendering(z_near, z_far)
    setup_dual_rendering(z_near, z_far, output_path, depth_output_prefix, rgb_output_prefix)

    # Set the camera output size
    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128

    #import_mesh('nozzle_5222583.stl')
    mesh = import_mesh(mesh_file)
    #mesh.location = (0.0, 0.0, 0.05)
    mesh.rotation_mode = 'QUATERNION'

    set_material_recursive(mesh, create_shadeless_material(rgba=(0,1,0,1)))

    gripper = Gripper()
    gripper.set_material(create_shadeless_material(rgba=(1,0,0,1)))

    rng = Random(1)

    # Set the number of frames in our "animation".  Blender defaults
    # to starting on frame 1, we like 0-based indexing though.
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 2*num_pairs - 1

    q_file = open(quaternion_output_file, "w")

    # For each frame, generate a different keyframed rotation on the mesh
    for fno in range(num_pairs):
        bpy.context.scene.frame_current = 2*fno
        q1 = random_quaternion(rng)
        mesh.rotation_quaternion = q1
        mesh.keyframe_insert(data_path='rotation_quaternion', frame=2*fno)


        bpy.context.scene.frame_current = 2*fno + 1
        q2 = q1.copy()
        q_rotate = random_quaternion_uniform30(rng)
        q2.rotate(q_rotate)
        mesh.rotation_quaternion = q2
        q_list = (q_rotate.x, q_rotate.y, q_rotate.z, q_rotate.w)
        q_file.write("%.16f %.16f %.16f %.16f\n" % q_list)
        mesh.keyframe_insert(data_path='rotation_quaternion', frame= 2*fno + 1)


    q_file.close()

    # reset to the first frame
    bpy.context.scene.frame_current = 0

    # Set up rendering
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NOTE: THESE SETTINGS ARE IGNORED AFTER setup_dual_rendering(...)
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.filepath = depth_output_prefix
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Start the rendering
    bpy.ops.render.render(animation=True)

    #bpy.ops.wm.quit_blender()
