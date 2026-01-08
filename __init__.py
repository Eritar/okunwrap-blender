# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "OKUnwrap",
    "author": "Iurii Kotlov, Vladislav Lopanov",
    "description": "Automatic curvature-based UV unwrapping",
    "blender": (4, 23, 0),
    "version": (1, 1, 0),
    "location": "",
    "warning": "",
    "category": "Generic",
}

import bpy
import bmesh
import time
import numpy as np
from numpy.linalg import norm
from mathutils import Vector
from math import degrees, acos
from ctypes import *
import sys
import addon_utils
import pathlib
import platform
import traceback
from . import okunwrap_core

okunwrap_dll = None


def getLibraryPath():
    match platform.system():
        case "Windows":
            return "okunwrap.dll"
        case "Darwin":
            return "libokunwrap.dylib"
        case _:
            return "libokunwrap.so"


class ExternalArray(Structure):
    _fields_ = [("data", c_void_p), ("size", c_int)]


class VertexData(ExternalArray):
    pass


class EdgeData(ExternalArray):
    pass


class SeamData(ExternalArray):
    pass


class CurvatureArray(ExternalArray):
    pass


class Edge(Structure):
    _fields_ = [("v1", c_int), ("v2", c_int)]


class EdgeMapData(Structure):
    _fields_ = [("data", POINTER(c_int)), ("size", c_int)]


class SeamData(Structure):
    _fields_ = [
        ("data", POINTER(c_bool)),
        ("size", c_int),
    ]


class UnwrapSettings(Structure):
    _fields_ = [
        ("biasCurvatureAmount", c_float),
        ("biasInlineLoopnessAmount", c_float),
        ("biasSeamClosenessAmount", c_float),
        ("curvatureThreshold", c_float),
        ("seamMarginAmount", c_int),
        ("seamSearchRadius", c_int),
        ("extendAmount", c_int),
        ("unwrapSteps", c_int),
    ]


def prepareMeshData(unwrap_batch, bm, obj_data):
    edges = bm.edges
    verts = bm.verts
    arr_edges = np.zeros(len(edges), dtype=np.dtype("i,i,i,b,f", align=True))

    for idx, edge in enumerate(edges):
        edge_angle = 0

        if edge.is_contiguous:
            edge_angle = edge.calc_face_angle()
        else:
            if edge.is_boundary:
                edge.seam = True

        u = edge.verts[0]
        v = edge.verts[1]
        arr_edges[idx] = (edge.index, u.index, v.index, edge.seam, edge_angle)

    okunwrap_dll.OKUnwrap_Batch_InitMesh(
        unwrap_batch,
        EdgeData(arr_edges.ctypes._as_parameter_, len(arr_edges)),
        VertexData(obj_data.vertices[0].as_pointer(), len(verts)),
    )

    return arr_edges


def beginUnwrapBatch(bm, obj_data, context, post_process=False):
    CURVATURE_Properties = context.scene.CURVATURE_Properties

    if not post_process:
        batch = okunwrap_dll.OKUnwrap_Batch_Begin(
            UnwrapSettings(
                CURVATURE_Properties.biasCurvatureAmount,
                CURVATURE_Properties.biasInlineLoopnessAmount,
                CURVATURE_Properties.biasSeamClosenessAmount,
                CURVATURE_Properties.curvatureThreshold,
                CURVATURE_Properties.seamMarginAmount,
                CURVATURE_Properties.seamSearchRadius,
                CURVATURE_Properties.extendAmount,
                CURVATURE_Properties.unwrapSteps,
            ),
        )
    else:
        batch = okunwrap_dll.OKUnwrap_Batch_Begin(
            UnwrapSettings(
                CURVATURE_Properties.biasCurvatureAmount,
                CURVATURE_Properties.biasInlineLoopnessAmount,
                5,
                0.01,
                1,
                CURVATURE_Properties.seamSearchRadius,
                CURVATURE_Properties.extendAmount,
                CURVATURE_Properties.unwrapSteps,
            ),
        )
    prepareMeshData(batch, bm, obj_data)

    return batch


def endUnwrapBatch(batch):
    if batch is None:
        return

    okunwrap_dll.OKUnwrap_Batch_End(batch)


def loadLibrary():
    global okunwrap_dll

    root_folder = pathlib.Path(__file__).parent.resolve()
    print(root_folder, sys.path)
    # sys.path.insert(0, root_folder)
    import os

    # os.add_dll_directory(root_folder)

    # print(okunwrap_core.add((1, 1, 1), (2, 2, 2)))

    # import example_module

    # addon_path = None
    # for mod in addon_utils.modules():
    #     if mod.bl_info.get("name") != "OKUnwrap":
    #         continue
    #     p = pathlib.Path(mod.__file__)
    #     addon_path = p.parent / getLibraryPath()
    # okunwrap_dll = CDLL(str(addon_path))

    # try:
    #     okunwrap_dll.OKUnwrap_Batch_Begin.argtypes = [UnwrapSettings]
    #     okunwrap_dll.OKUnwrap_Batch_Begin.restype = c_void_p

    #     okunwrap_dll.OKUnwrap_Batch_End.argtypes = [c_void_p]
    #     okunwrap_dll.OKUnwrap_Batch_End.restype = None

    #     okunwrap_dll.OKUnwrap_Batch_InitMesh.argtypes = [c_void_p, EdgeData, VertexData]
    #     okunwrap_dll.OKUnwrap_Batch_InitMesh.restype = c_bool

    #     okunwrap_dll.OKUnwrap_Batch_Execute.argtypes = [c_void_p]
    #     okunwrap_dll.OKUnwrap_Batch_Execute.restype = c_bool

    #     okunwrap_dll.OKUnwrap_Batch_Result.argtypes = [c_void_p]
    #     okunwrap_dll.OKUnwrap_Batch_Result.restype = POINTER(c_int)
    # except Exception as e:
    #     print(f"OKUnwrap register(): {e}", file=sys.stderr)
    # finally:
    #     pass


def unloadLibrary():
    import _ctypes

    # try:
    #     if platform.system() == "Windows":
    #         _ctypes.FreeLibrary(okunwrap_dll._handle)
    #     else:
    #         _ctypes.dlclose(okunwrap_dll._handle)
    # except Exception as e:
    #     print(f"OKUnwrap unregister(): {e}", file=sys.stderr)


def selectDistortedFaces(context, obj, bm, threshold):
    """
    Select faces with UV angular distortion above a normalized threshold.
    """
    # threshold = 0.2

    uv_layer = bm.loops.layers.uv.active

    face_distortions = {}
    max_distortion = 0.0

    for face in bm.faces:
        total_distortion = 0.0
        loops = face.loops
        num_loops = len(loops)

        for i in range(num_loops):
            # Get 3D angle
            geom_angle = loops[i].calc_angle()
            # TODO: Use curvature values

            # Get UV coordinates for the current corner and its neighbors
            uv_center = loops[i][uv_layer].uv
            uv_prev = loops[i - 1][uv_layer].uv
            uv_next = loops[(i + 1) % num_loops][uv_layer].uv

            # Calculate UV angle
            uv_angle = get_uv_angle(uv_center, uv_prev, uv_next)

            # Accumulate distortion
            total_distortion += abs(geom_angle - uv_angle)

        face_distortions[face.index] = total_distortion
        if total_distortion > max_distortion:
            max_distortion = total_distortion

    # Normalize distortions and select faces
    bpy.ops.mesh.select_all(action="DESELECT")

    if max_distortion > 0:
        for face in bm.faces:
            normalized_distortion = face_distortions[face.index] / max_distortion
            if normalized_distortion > threshold:
                face.select = True
    bpy.ops.uv.select_linked()

    distortedFaces = [f for f in bm.faces if f.select]
    return distortedFaces


def selectDistortedEdges(context, obj, bm, threshold):
    """
    Select edges with UV length distortion above a normalized threshold.
    """
    # threshold = 0.8

    uv_layer = bm.loops.layers.uv.active

    edge_distortions = {}
    max_distortion = 0.0

    for edge in bm.edges:
        geom_length = edge.calc_length()
        if geom_length < 0.0001:
            continue

        max_edge_uv_distortion = 0.0

        # An edge can be part of multiple faces, and thus have multiple UV representations
        for face in edge.link_faces:
            uv1, uv2 = None, None
            # Find the UV coordinates corresponding to the edge's vertices within this face
            for loop in face.loops:
                if loop.vert == edge.verts[0]:
                    uv1 = loop[uv_layer].uv
                elif loop.vert == edge.verts[1]:
                    uv2 = loop[uv_layer].uv

            if uv1 and uv2:
                uv_length = (uv1 - uv2).length
                # Use the absolute difference from a perfect 1.0 ratio
                distortion = abs(1.0 - (uv_length / geom_length))
                if distortion > max_edge_uv_distortion:
                    max_edge_uv_distortion = distortion

        edge_distortions[edge.index] = max_edge_uv_distortion
        if max_edge_uv_distortion > max_distortion:
            max_distortion = max_edge_uv_distortion

    # Set selection mode to edges
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")

    edge_distortions = {
        k: v
        for k, v in sorted(edge_distortions.items(), key=lambda x: x[1], reverse=True)
    }
    # print(edge_distortions)
    # bm.edges[next(iter(edge_distortions))].select = True

    if max_distortion > 0:
        for edge in bm.edges:
            normalized_distortion = edge_distortions[edge.index] / max_distortion
            # bpy.ops.mesh.select_all(action='DESELECT')
            if normalized_distortion > threshold:
                edge.select = True
                bpy.ops.mesh.py_extend_to_loop()
                edge.select = False


def get_uv_angle(uv1, uv2, uv3):
    """Calculate the angle between three 2D UV coordinates."""
    vec1 = uv2 - uv1
    vec2 = uv3 - uv1

    # Handle cases where vectors have zero length
    if vec1.length == 0 or vec2.length == 0:
        return 0.0

    dot_product = vec1.dot(vec2)

    # Clamp the dot product to avoid math domain errors
    clamp_dot = max(-1.0, min(1.0, dot_product / (vec1.length * vec2.length)))

    return acos(clamp_dot)


# Code is absolutely gross, but such is life
def remove_loop(start_edge: bmesh.types.BMEdge, extendAmount):
    removedEdges = list()
    if start_edge.seam == False:
        closeEdges = {start_edge for start_edge in start_edge.verts[0].link_edges} | {
            start_edge for start_edge in start_edge.verts[1].link_edges
        }

        for e in closeEdges:
            if e.seam == True:
                start_edge = e
                removedEdges.append(e)
                break
    else:
        removedEdges.append(start_edge)

    for a in range(2):
        edgeToCheck = start_edge

        tempVert = edgeToCheck.verts[a]
        endBranch = False

        for i in range(extendAmount):
            if endBranch:
                break

            neighboringEdges = set(tempVert.link_edges)
            neighboringEdges.discard(edgeToCheck)

            for e in neighboringEdges:
                seamsCloseBy = 0

                for temp_e in tempVert.link_edges:
                    if temp_e == edgeToCheck:
                        continue
                    if temp_e.seam == True:
                        seamsCloseBy += 1

                if seamsCloseBy >= 2 or seamsCloseBy == 0:
                    endBranch = True
                    break

                if e.seam == True:
                    tempVert = e.other_vert(
                        (set(e.verts) & set(edgeToCheck.verts)).pop()
                    )
                    edgeToCheck = e
                    removedEdges.append(e)
                    e.seam = False
                    continue

    start_edge.seam = False
    return removedEdges


class MESH_OT_unwrap(bpy.types.Operator):
    """Unwrap selected/edited meshes using the settings below"""

    bl_idname = "mesh.unwrap"
    bl_label = "unwrap"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        CURVATURE_Properties = context.scene.CURVATURE_Properties
        start = time.perf_counter()

        try:
            for obj in bpy.context.selected_objects:
                if obj.type != "MESH":
                    continue

                obj_data = obj.data

                # if not obj.mode == "EDIT":
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.mode_set(mode="EDIT")

                bm = bmesh.from_edit_mesh(obj_data)

                bm.edges.ensure_lookup_table()

                if len(obj_data.vertices) != len(bm.verts) or len(
                    obj_data.edges
                ) != len(bm.edges):
                    print(
                        f"Verts and edges count mismatch between object data and BMesh! {len(obj_data.vertices)=}/{len(bm.verts)=}, {len(obj_data.edges)=}/{len(bm.edges)=}",
                        file=sys.stderr,
                    )
                    continue

                # continue

                if CURVATURE_Properties.overwriteSeams:
                    for edge in bm.edges:
                        edge.seam = False

                if CURVATURE_Properties.useSharpAsSeams:
                    for edge in bm.edges:
                        if not edge.smooth:
                            edge.seam = True

                arr_verts = np.ctypeslib.as_array(
                    cast(obj_data.vertices[0].as_pointer(), POINTER(c_float)),
                    (len(obj_data.vertices) * 3,),
                ).view(dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")]))
                arr_edges = np.ctypeslib.as_array(
                    cast(obj_data.edges[0].as_pointer(), POINTER(c_int)),
                    (len(obj_data.edges) * 2,),
                ).view(dtype=np.dtype([("u", "i"), ("v", "i")]))

                curvatures = np.zeros(len(obj_data.edges), dtype=np.double)

                for idx, edge in enumerate(bm.edges):
                    if edge.is_contiguous:
                        curvatures[edge.index] = edge.calc_face_angle()
                    # else:
                    #     if edge.is_boundary:
                    #         edge.seam = True

                seams = np.empty(len(obj_data.edges), dtype=bool)
                selected_edge = None

                obj_data.edges.foreach_get("use_seam", seams)

                for edge in bm.edges:
                    if edge.select:
                        selected_edge = edge.index

                cpp_mesh = okunwrap_core.Mesh(
                    arr_verts,
                    arr_edges,
                    seams,
                    curvatures,
                )
                op = okunwrap_core.ExtendToLoopOperation(
                    cpp_mesh,
                    selected_edge,
                    okunwrap_core.UnwrapSettings(
                        CURVATURE_Properties.biasCurvatureAmount,
                        CURVATURE_Properties.biasInlineLoopnessAmount,
                        CURVATURE_Properties.biasSeamClosenessAmount,
                        CURVATURE_Properties.curvatureThreshold,
                        CURVATURE_Properties.seamMarginAmount,
                        CURVATURE_Properties.seamSearchRadius,
                        CURVATURE_Properties.extendAmount,
                        CURVATURE_Properties.unwrapSteps,
                    ),
                )
                op.execute()
                cpp_mesh.notify_need_update()

                # batch = beginUnwrapBatch(bm, obj_data, context)
                # okunwrap_dll.OKUnwrap_Batch_Execute(batch)

                # ptr_result = okunwrap_dll.OKUnwrap_Batch_Result(batch)

                for i, edge in enumerate(bm.edges):
                    edge.seam = seams[i]

                if CURVATURE_Properties.enablePostProcess:
                    print("\n")
                    bpy.ops.mesh.select_all(action="SELECT")

                    bpy.ops.uv.unwrap(method="CONFORMAL")

                    distortedFaces = selectDistortedFaces(
                        context,
                        obj,
                        bm,
                        CURVATURE_Properties.postProcessDistortionThreshold,
                    )
                    distortedEdges = list()
                    for f in distortedFaces:
                        for e in f.edges:
                            distortedEdges.append(e)
                    distortedEdges = set(distortedEdges)
                    distortedEdgesIndices = {e.index for e in distortedEdges}

                    smallBatch = beginUnwrapBatch(bm, obj_data, context, True)
                    okunwrap_dll.OKUnwrap_Batch_Execute(smallBatch)

                    small_ptr_result = okunwrap_dll.OKUnwrap_Batch_Result(smallBatch)

                    for i, edge in enumerate(bm.edges):
                        if edge.index in distortedEdgesIndices:
                            edge.seam = bool(small_ptr_result[i])

                    bmesh.update_edit_mesh(obj_data)

                    # bpy.ops.mesh.select_all(action='DESELECT')

            endUnwrapBatch(batch)
            endUnwrapBatch(smallBatch)

            if CURVATURE_Properties.unwrapAtEnd:
                bpy.ops.mesh.select_all(action="SELECT")
                if bpy.app.version >= (4, 3, 0):
                    bpy.ops.uv.unwrap(method=CURVATURE_Properties.unwrapType)
                else:
                    bpy.ops.uv.unwrap()
                bpy.ops.mesh.select_all(action="DESELECT")

            VIEW3D_PT_OKUnwrap.operation_time = round(
                (time.perf_counter() - start) * 1000, 2
            )
        except Exception as e:
            print(f"MESH_OT_unwrap: {traceback.format_exc()}", file=sys.stderr)
        finally:
            bmesh.update_edit_mesh(obj_data)
            # endUnwrapBatch(batch)
        print("====================================================")

        return {"FINISHED"}


class MESH_OT_unload_dll(bpy.types.Operator):
    """Unload OKUnwrap DLL"""

    bl_idname = "mesh.unload_dll"
    bl_label = "PLACEHOLDER"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        unloadLibrary()
        return {"FINISHED"}


class MESH_OT_load_dll(bpy.types.Operator):
    """Load OKUnwrap DLL"""

    bl_idname = "mesh.load_dll"
    bl_label = "PLACEHOLDER"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        global okunwrap_dll
        okunwrap_dll = CDLL(getLibraryPath())

        return {"FINISHED"}


class MESH_OT_create_uv_loop(bpy.types.Operator):
    """Create seam loops from selected edges, using settings from the main unwrap function"""

    bl_idname = "mesh.create_uv_loop"
    bl_label = "create_uv_loop"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        CURVATURE_Properties = context.scene.CURVATURE_Properties
        start = time.perf_counter()

        for obj in bpy.context.selected_objects:
            if obj.type != "MESH":
                continue

            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.mode_set(mode="EDIT")

            obj_data = obj.data
            # initialMeshSelectModeState = bpy.context.tool_settings.mesh_select_mode[:]

            bm = bmesh.from_edit_mesh(obj_data)
            # bm.select_mode = {"VERT", "EDGE", "FACE"}
            # bm.select_flush_mode()
            # context.tool_settings.mesh_select_mode = (False, True, False)

            selected_edges = {edge for edge in bm.edges if edge.select}
            arr_verts = np.ctypeslib.as_array(
                cast(obj_data.vertices[0].as_pointer(), POINTER(c_float)),
                (len(obj_data.vertices) * 3,),
            ).view(dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")]))
            arr_edges = np.ctypeslib.as_array(
                cast(obj_data.edges[0].as_pointer(), POINTER(c_int)),
                (len(obj_data.edges) * 2,),
            ).view(dtype=np.dtype([("u", "i"), ("v", "i")]))

            curvatures = np.zeros(len(obj_data.edges), dtype=np.double)

            for idx, edge in enumerate(bm.edges):
                if edge.is_contiguous:
                    curvatures[edge.index] = edge.calc_face_angle()

            seams = np.empty(len(obj_data.edges), dtype=bool)

            obj_data.edges.foreach_get("use_seam", seams)

            for edge in bm.edges:
                if edge.select:
                    selected_edge = edge.index

            cpp_mesh = okunwrap_core.Mesh(
                arr_verts,
                arr_edges,
                seams,
                curvatures,
            )
            for e in selected_edges:
                # remove_loop(e, CURVATURE_Properties.extendAmount)
                op = okunwrap_core.ExtendToLoopOperation(
                    cpp_mesh,
                    e.index,
                    okunwrap_core.UnwrapSettings(
                        CURVATURE_Properties.biasCurvatureAmount,
                        CURVATURE_Properties.biasInlineLoopnessAmount,
                        CURVATURE_Properties.biasSeamClosenessAmount,
                        CURVATURE_Properties.curvatureThreshold,
                        CURVATURE_Properties.seamMarginAmount,
                        CURVATURE_Properties.seamSearchRadius,
                        CURVATURE_Properties.extendAmount,
                        CURVATURE_Properties.unwrapSteps,
                    ),
                )
                op.execute()
            cpp_mesh.notify_need_update()

            # batch = beginUnwrapBatch(bm, obj_data, context)
            # okunwrap_dll.OKUnwrap_Batch_Execute(batch)

            # ptr_result = okunwrap_dll.OKUnwrap_Batch_Result(batch)

            for i, edge in enumerate(bm.edges):
                edge.seam = seams[i]

            if len(bpy.context.selected_objects) == 1 and len(selected_edges) == 0:
                self.report({"INFO"}, "No edges selected")
                return {"CANCELLED"}

            # bpy.ops.mesh.select_all(action='DESELECT')
            # context.tool_settings.mesh_select_mode = initialMeshSelectModeState
            bmesh.update_edit_mesh(obj_data)

        VIEW3D_PT_OKUnwrap.operation_time = round(
            (time.perf_counter() - start) * 1000, 2
        )

        return {"FINISHED"}


class MESH_OT_remove_uv_loop(bpy.types.Operator):
    """Remove seam loops that are a part of/adjacent to selection. Selection will convert to edges automatically"""

    bl_idname = "mesh.remove_uv_loop"
    bl_label = "remove_uv_loop"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        CURVATURE_Properties = context.scene.CURVATURE_Properties
        start = time.perf_counter()

        # removedLoops = list()

        for obj in bpy.context.selected_objects:
            if obj.type != "MESH":
                continue

            obj_data = obj.data

            if not obj.mode == "EDIT":
                bpy.ops.object.mode_set(mode="EDIT")
            initialMeshSelectModeState = bpy.context.tool_settings.mesh_select_mode[:]

            bm = bmesh.from_edit_mesh(obj_data)
            bm.select_mode = {"VERT", "EDGE", "FACE"}
            bm.select_flush_mode()
            context.tool_settings.mesh_select_mode = (False, True, False)

            selected_edges = {edge for edge in bm.edges if edge.select}

            if selected_edges:
                for e in selected_edges:
                    remove_loop(e, CURVATURE_Properties.extendAmount)
                    # removedLoops.extend(remove_loop(e, CURVATURE_Properties.extendAmount))

            if len(bpy.context.selected_objects) == 1 and len(selected_edges) == 0:
                self.report({"INFO"}, "No edges selected")
                return {"CANCELLED"}

            # bpy.ops.mesh.select_all(action='DESELECT')
            context.tool_settings.mesh_select_mode = initialMeshSelectModeState
            bmesh.update_edit_mesh(obj_data)

        VIEW3D_PT_OKUnwrap.operation_time = round(
            (time.perf_counter() - start) * 1000, 2
        )

        return {"FINISHED"}


class VIEW3D_PT_OKUnwrap(bpy.types.Panel):  # class naming convention ‘CATEGORY_PT_name’

    # where to add the panel in the UI
    bl_space_type = "VIEW_3D"  # 3D Viewport area (find list of values here https://docs.blender.org/api/current/bpy_types_enum_items/space_type_items.html#rna-enum-space-type-items)
    bl_region_type = "UI"  # Sidebar region (find list of values here https://docs.blender.org/api/current/bpy_types_enum_items/region_type_items.html#rna-enum-region-type-items)

    # add labels
    bl_category = "OKUnwrap"  # found in the Sidebar
    bl_label = "OKUnwrap"  # found at the top of the Panel

    operation_time = 0

    def draw(self, context):
        CURVATURE_Properties = context.scene.CURVATURE_Properties

        row = self.layout.row()
        row.operator("mesh.unwrap", text="Unwrap")
        self.layout.separator()

        row = self.layout.row()
        row.operator("mesh.create_uv_loop", text="Create Loops")
        row.operator("mesh.remove_uv_loop", text="Remove Loops")
        # row.operator("mesh.remove_uv_loop", text="", icon='TRASH')
        self.layout.separator()

        row = self.layout.row()
        row.operator("mesh.load_dll", text="LOAD DLL")
        row = self.layout.row()
        row.operator("mesh.unload_dll", text="UNLOAD DLL")

        self.layout.separator()
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "unwrapSteps")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "curvatureThreshold")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "extendAmount")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "seamMarginAmount")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "seamSearchRadius")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "overwriteSeams")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "useSharpAsSeams")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "unwrapAtEnd", text="Unwrap UV")

        if CURVATURE_Properties.unwrapAtEnd:
            if bpy.app.version >= (4, 3, 0):
                row = self.layout.row(heading="Unwrap Type:")
                row.prop(CURVATURE_Properties, "unwrapType", text="")
                # self.layout.separator()

        row = self.layout.row()
        row.prop(CURVATURE_Properties, "enablePostProcess")
        if CURVATURE_Properties.enablePostProcess:
            row = self.layout.row()
            row.prop(CURVATURE_Properties, "postProcessDistortionThreshold")

        self.layout.separator()
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "biasCurvatureAmount")
        row = self.layout.row()
        row.prop(
            CURVATURE_Properties,
            "biasInlineLoopnessAmount",
            text="Inline Loopness Bias",
        )
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "biasSeamClosenessAmount")
        row = self.layout.row()

        self.layout.separator()
        self.layout.label(text=f"Operation time: {self.operation_time}ms")
        # row = self.layout.row()
        self.layout.label(text="Thank you so much for your support! ❤️️")


class CURVATURE_Properties(bpy.types.PropertyGroup):
    unwrapSteps: bpy.props.IntProperty(
        name="Unwrap Steps",
        default=30,
        min=1,
        soft_max=250,
        step=1,
        description="Unwrap iteration step. If result gives not enough seams, increase this value.\nRanges from 1 to 250+",
    )  # type: ignore

    curvatureThreshold: bpy.props.FloatProperty(
        name="Curvature Threshold",
        default=0.7,
        soft_min=0.01,
        max=1,
        step=0.02,
        description="Cutoff of curvature values that won't be considered. Higher the number - sharper features will be marked as seams. Generally, more high poly the model - lower this value should be.\nRanges from 0.01 to 1",
    )  # type: ignore

    seamMarginAmount: bpy.props.IntProperty(
        name="Seam Margin Amount",
        default=2,
        min=0,
        soft_max=10,
        step=1,
        description="Amount of margin every seam has, to avoid several seam loops closeby. More generally is better on dense geometry. If your lowpoly geo doesn't have enough seams - this value may be too high.\nRanges from 1 to 10+",
    )  # type: ignore

    seamSearchRadius: bpy.props.IntProperty(
        name="Seam Search Radius",
        default=3,
        min=0,
        soft_max=5,
        description="Radius in which seam loop will try to find other seams to connect with. Higher - more closely joined seams.\nValues higher than 4-5 increase timings significantly!\nRanges from 1 to 5+",
    )  # type: ignore

    overwriteSeams: bpy.props.BoolProperty(
        name="Overwrite Seams",
        default=True,
        description="Option to overwrite existing seams. If unchecked - OKUnwrap will add seams to already existing ones",
    )  # type: ignore

    unwrapAtEnd: bpy.props.BoolProperty(
        name="UV Unwrap",
        default=False,
        description="Will unwrap the model after seam generation. If unchecked - OKUnwrap will only add seams to the model",
    )  # type: ignore

    extendAmount: bpy.props.IntProperty(
        name="Extend Amount",
        default=500,
        min=0,
        soft_max=1000,
        step=1,
        description="Max length of each seam loop, generally around 500 is good, tweak this value if a lot of noise is present.\nRanges from 0 to 1000+",
    )  # type: ignore

    biasCurvatureAmount: bpy.props.FloatProperty(
        name="Curvature Bias",
        default=1,
        min=0,
        soft_max=1,
        step=0.01,
        description="Affects how much the curvature of the mesh will affect the seam loops. Higher the value - the more seams will adhere to sharp (both convex and concave) features.\nRanges from 0 to 1+",
    )  # type: ignore

    biasInlineLoopnessAmount: bpy.props.FloatProperty(
        name="Inline Loopness Bias",
        default=1,
        min=0,
        soft_max=1,
        step=0.01,
        description="Affects how much the seam loops will try to run straight.\nRanges from 0 to 1+",
    )  # type: ignore

    biasSeamClosenessAmount: bpy.props.FloatProperty(
        name="Seam Closeness Bias",
        default=0.6,
        min=0,
        soft_max=1,
        step=0.01,
        description="Affects how much the seam loops will try to connect with already existing seams.\nRanges from 0 to 1+",
    )  # type: ignore

    unwrapType: bpy.props.EnumProperty(
        name="Unwrap Type",
        default="ANGLE_BASED",
        items=[
            ("ANGLE_BASED", "Angle Based", ""),
            ("CONFORMAL", "Conformal", ""),
            ("MINIMUM_STRETCH", "Minimum Stretch", ""),
        ],
        description="Unwrap type - Angle Based, Conformal or Minimum Stretch",
    )  # type: ignore

    useSharpAsSeams: bpy.props.BoolProperty(
        name="Mark Sharp Edges as Seams",
        default=True,
        description="Option will mark all sharp edges as seams before running the algorithm",
    )  # type: ignore

    enablePostProcess: bpy.props.BoolProperty(
        name="Enable Post Process",
        default=True,
        description="Option will search for distorted uv islands and attempt to fix them",
    )  # type: ignore

    postProcessDistortionThreshold: bpy.props.FloatProperty(
        name="Post Process Threshold",
        default=0.6,
        min=0,
        soft_max=1,
        step=0.01,
        description="Threshold after which a UV island is considered distorted, and sent to post-processing",
    )  # type: ignore


classes = [
    VIEW3D_PT_OKUnwrap,
    MESH_OT_unwrap,
    MESH_OT_unload_dll,
    MESH_OT_load_dll,
    MESH_OT_create_uv_loop,
    MESH_OT_remove_uv_loop,
    CURVATURE_Properties,
]


# register the panel with Blender
def register():
    loadLibrary()

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.CURVATURE_Properties = bpy.props.PointerProperty(
        type=CURVATURE_Properties
    )


def unregister():
    unloadLibrary()

    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.CURVATURE_Properties


if __name__ == "__main__":
    register()
