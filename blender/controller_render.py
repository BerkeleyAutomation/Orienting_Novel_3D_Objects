#!/usr/bin/env blender --python

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) UC Berkeley AUTOLAB
#
# This file loads a mesh and renders depth map views of it using
# Blender.  Tested on Blender 2.82.  Most likely not compatible with
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
from random import Random
from mathutils import Quaternion
from math import pi, sin, cos, sqrt
import argparse
import pickle

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
    
def parse_args():
    """Parse arguments from the command line.
    -config to input your own yaml config file. Default is unsup_rbt_train_quat.yaml
    --pair to generate a pair of start and goal images
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', action='store_true')
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..',
                                           'cfg/tools/unsup_rbt_train_quat.yaml')
    parser.add_argument('-config', type=str, default=default_config_filename)
    args = parser.parse_args()
    return args

if "__main__" == __name__:
    # Delete the starting cube
    bpy.ops.object.delete(use_global=False)

    # ======================================== Options
    # Set these values to specify the depth range of the camera
    z_near = 0.83
    z_far = 1.035
    # Number of frames to generate
    num_pairs = 10
    # Input mesh
    # mesh_file = 'hex_vase_3276254.stl'
    # mesh_file = 'banana_3dnet/banana_3dnet.obj'
    mesh_file = 'elephant_1_3dnet/elephant_1_3dnet.obj'
    # Output prefix
    cur_quaternion_output_file = os.path.join(os.getcwd(), "blender/cur_quat.p")
    true_quaternion_output_file = os.path.join(os.getcwd(), "blender/true_quat.p")
    rot_quaternion_output_file = os.path.join(os.getcwd(), "blender/rot_quat.p")
    depth_output_prefix = os.path.join(os.getcwd(), "blender/image_")
    # ======================================== End of Options
    args = parse_args()

    # position the camera 1 meter above the view plane.
    # Objects at the origin will be centered in the rendered view
    camera = bpy.data.objects['Camera']
    camera.location = (0.0, 0.0, 1.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.clip_start = z_near
    camera.data.clip_end = z_far
    # camera.data.angle = 27.0*pi/180.0
    camera.data.angle = 9.0*pi/180.0

    setup_depth_rendering(z_near, z_far)

    # Set the camera output size
    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
    
    mesh = import_mesh(mesh_file)
    mesh.rotation_mode = 'QUATERNION'
    
    rng = Random(1)

    if args.pair:
        # Set the number of frames in our "animation".  Blender defaults
        # to starting on frame 1, we like 0-based indexing though.
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 1
        
        # For each frame, generate a different keyframed rotation on the mesh
        bpy.context.scene.frame_current = 0
        q1 = random_quaternion(rng)
        mesh.rotation_quaternion = q1
        mesh.keyframe_insert(data_path='rotation_quaternion', frame=0)

        bpy.context.scene.frame_current = 1
        q2 = q1.copy()
        q_rotate = random_quaternion_uniform30(rng)
        q2.rotate(q_rotate)
        mesh.rotation_quaternion = q2
        cur_quat, true_quat, rot_quat = [[q1.x, q1.y, q1.z, q1.w],
            [q2.x, q2.y, q2.z, q2.w],
            [q_rotate.x, q_rotate.y, q_rotate.z, q_rotate.w]]

        pickle.dump(cur_quat, open(cur_quaternion_output_file, "wb"))
        pickle.dump(true_quat, open(true_quaternion_output_file, "wb"))
        pickle.dump(rot_quat, open(rot_quaternion_output_file, "wb"))
        # q_file.write("%.16f %.16f %.16f %.16f\n" % q_list)
        mesh.keyframe_insert(data_path='rotation_quaternion', frame= 1)
    else:
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0

        cur_quat = pickle.load(cur_quat, open(cur_quaternion_output_file, "rb"))
        true_quat = pickle.load(true_quat, open(true_quaternion_output_file, "rb"))
        rot_quat = pickle.load(rot_quat, open(rot_quaternion_output_file, "rb"))
        pred_quat = pickle.load(pred_quat, open(rot_quaternion_output_file, "rb"))

        bpy.context.scene.frame_current = 0

        cur_quat = Quaternion((cur_quat[3],cur_quat[0],cur_quat[1],cur_quat[2]))
        true_quat = Quaternion((true_quat[3],true_quat[0],true_quat[1],true_quat[2]))
        rot_quat = Quaternion((rot_quat[3],rot_quat[0],rot_quat[1],rot_quat[2]))
        pred_quat = Quaternion((pred_quat[3],pred_quat[0],pred_quat[1],pred_quat[2]))

        next_quat = cur_quat.copy()
        next_quat.rotate(pred_quat)
        next_quat = cur_quat.slerp(next_quat, 0.3)
        mesh.rotation_quaternion = next_quat
        q_list = [next_quat.x, next_quat.y, next_quat.z, next_quat.w]
        pickle.dump(q_list, open(cur_quaternion_output_file, "wb"))
        mesh.keyframe_insert(data_path='rotation_quaternion', frame= 0)

    # reset to the first frame
    bpy.context.scene.frame_current = 0

    # Set up rendering
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.filepath = depth_output_prefix

    # Start the rendering
    bpy.ops.render.render(animation=True)
    
    bpy.ops.wm.quit_blender()