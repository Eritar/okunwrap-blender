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
from ctypes import *
import sys
import traceback
from . import okunwrap_core


def loadLibrary():
    pass


def unloadLibrary():
    pass


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

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.mode_set(mode="EDIT")

                obj_data = obj.data
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

                curvatures = np.zeros(len(obj_data.edges), dtype=np.double)

                for idx, edge in enumerate(bm.edges):
                    if CURVATURE_Properties.overwriteSeams:
                        edge.seam = False

                    if CURVATURE_Properties.useSharpAsSeams:
                        if not edge.smooth:
                            edge.seam = True

                    if edge.is_contiguous:
                        curvatures[edge.index] = edge.calc_face_angle()
                    else:
                        if edge.is_boundary:
                            edge.seam = True

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.mode_set(mode="EDIT")

                obj_data = obj.data
                bm = bmesh.from_edit_mesh(obj_data)

                bm.edges.ensure_lookup_table()

                arr_verts = np.ctypeslib.as_array(
                    cast(obj_data.vertices[0].as_pointer(), POINTER(c_float)),
                    (len(obj_data.vertices) * 3,),
                ).view(dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")]))
                arr_edges = np.ctypeslib.as_array(
                    cast(obj_data.edges[0].as_pointer(), POINTER(c_int)),
                    (len(obj_data.edges) * 2,),
                ).view(dtype=np.dtype([("u", "i"), ("v", "i")]))
                seams = np.empty(len(obj_data.edges), dtype=bool)

                obj_data.edges.foreach_get("use_seam", seams)

                cpp_mesh = okunwrap_core.Mesh(
                    arr_verts,
                    arr_edges,
                    seams,
                    curvatures,
                )
                op = okunwrap_core.UnwrapOperation(
                    cpp_mesh,
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
                bpy.ops.object.mode_set(mode="OBJECT")
                obj_data.edges.foreach_set("use_seam", seams)
                bpy.ops.object.mode_set(mode="EDIT")
                bmesh.update_edit_mesh(obj_data, destructive=False)

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
            pass
        print("====================================================")

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
            bm = bmesh.from_edit_mesh(obj_data)

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

            for i, edge in enumerate(bm.edges):
                edge.seam = seams[i]

            if len(bpy.context.selected_objects) == 1 and len(selected_edges) == 0:
                self.report({"INFO"}, "No edges selected")
                return {"CANCELLED"}

            if CURVATURE_Properties.enableUnwrapAfterLoop:
                bpy.ops.mesh.select_all(action="SELECT")
                if bpy.app.version >= (4, 3, 0):
                    bpy.ops.uv.unwrap(method=CURVATURE_Properties.unwrapType)
                else:
                    bpy.ops.uv.unwrap()
                bpy.ops.mesh.select_all(action="DESELECT")

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

            if len(bpy.context.selected_objects) == 1 and len(selected_edges) == 0:
                self.report({"INFO"}, "No edges selected")
                return {"CANCELLED"}

            context.tool_settings.mesh_select_mode = initialMeshSelectModeState

            if CURVATURE_Properties.enableUnwrapAfterLoop:
                bpy.ops.mesh.select_all(action="SELECT")
                if bpy.app.version >= (4, 3, 0):
                    bpy.ops.uv.unwrap(method=CURVATURE_Properties.unwrapType)
                else:
                    bpy.ops.uv.unwrap()
                bpy.ops.mesh.select_all(action="DESELECT")

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
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "enableUnwrapAfterLoop")

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
        name="Overwrite Existing Seams",
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
        description="Blender's built in unwrap after generating seams - Angle Based, Conformal or Minimum Stretch",
    )  # type: ignore

    useSharpAsSeams: bpy.props.BoolProperty(
        name="Mark Sharp Edges as Seams",
        default=True,
        description="Option will mark all sharp edges as seams before running the algorithm",
    )  # type: ignore

    enableUnwrapAfterLoop: bpy.props.BoolProperty(
        name="Unwrap After Manual Loops",
        default=True,
        description="Blender unwrap the mesh after every manual Create Loop/Remove Loop",
    )  # type: ignore


classes = [
    VIEW3D_PT_OKUnwrap,
    MESH_OT_unwrap,
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
