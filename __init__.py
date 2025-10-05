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
    "version": (1, 0, 1),
    "location": "",
    "warning": "",
    "category": "Generic",
}

import bpy
import bmesh
import time
import numpy as np
from numpy.linalg import norm
from ctypes import *
import sys
import addon_utils
import pathlib
import platform

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
    arr_edges = np.zeros(len(bm.edges), dtype=np.dtype("i,i,i,b,f", align=True))

    for idx, edge in enumerate(bm.edges):
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
        VertexData(obj_data.vertices[0].as_pointer(), len(bm.verts)),
    )

    return arr_edges


def beginUnwrapBatch(bm, obj_data, context):
    CURVATURE_Properties = context.scene.CURVATURE_Properties

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
    prepareMeshData(batch, bm, obj_data)

    return batch


def endUnwrapBatch(batch):
    if batch is None:
        return

    okunwrap_dll.OKUnwrap_Batch_End(batch)


def loadLibrary():
    global okunwrap_dll

    addon_path = None
    for mod in addon_utils.modules():
        if mod.bl_info.get("name") != "OKUnwrap":
            continue
        p = pathlib.Path(mod.__file__)
        addon_path = p.parent / getLibraryPath()
    okunwrap_dll = CDLL(str(addon_path))

    try:
        okunwrap_dll.OKUnwrap_Batch_Begin.argtypes = [UnwrapSettings]
        okunwrap_dll.OKUnwrap_Batch_Begin.restype = c_void_p

        okunwrap_dll.OKUnwrap_Batch_End.argtypes = [c_void_p]
        okunwrap_dll.OKUnwrap_Batch_End.restype = None

        okunwrap_dll.OKUnwrap_Batch_InitMesh.argtypes = [c_void_p, EdgeData, VertexData]
        okunwrap_dll.OKUnwrap_Batch_InitMesh.restype = c_bool

        okunwrap_dll.OKUnwrap_Batch_Execute.argtypes = [c_void_p]
        okunwrap_dll.OKUnwrap_Batch_Execute.restype = c_bool

        okunwrap_dll.OKUnwrap_Batch_Result.argtypes = [c_void_p]
        okunwrap_dll.OKUnwrap_Batch_Result.restype = POINTER(c_int)
    except Exception as e:
        print(f"OKUnwrap register(): {e}", file=sys.stderr)
    finally:
        pass


def unloadLibrary():
    import _ctypes

    try:
        if platform.system() == "Windows":
            _ctypes.FreeLibrary(okunwrap_dll._handle)
        else:
            _ctypes.dlclose(okunwrap_dll._handle)
    except Exception as e:
        print(f"OKUnwrap unregister(): {e}", file=sys.stderr)


class MESH_OT_unwrap(bpy.types.Operator):
    """Unwraps selected/edited meshes using the settings below"""

    bl_idname = "mesh.unwrap"
    bl_label = "unwrap"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        CURVATURE_Properties = context.scene.CURVATURE_Properties
        start = time.perf_counter()
        batch = None

        try:
            for obj in bpy.context.selected_objects:
                if obj.type != "MESH":
                    continue

                obj_data = obj.data

                if not obj.mode == "EDIT":
                    bpy.ops.object.mode_set(mode="EDIT")

                bm = bmesh.from_edit_mesh(obj_data)

                bm.edges.ensure_lookup_table()

                if CURVATURE_Properties.overwriteSeams:
                    for edge in bm.edges:
                        edge.seam = False

                if CURVATURE_Properties.useSharpAsSeams:
                    for edge in bm.edges:
                        if not edge.smooth:
                            edge.seam = True

                batch = beginUnwrapBatch(bm, obj_data, context)
                okunwrap_dll.OKUnwrap_Batch_Execute(batch)

                ptr_result = okunwrap_dll.OKUnwrap_Batch_Result(batch)

                for i, edge in enumerate(bm.edges):
                    edge.seam = ptr_result[i]
                

                bmesh.update_edit_mesh(obj_data)

            if CURVATURE_Properties.unwrapAtEnd:
                if bpy.app.version >= (4, 3, 0):
                    bpy.ops.uv.unwrap(method=CURVATURE_Properties.unwrapType)
                else:
                    bpy.ops.uv.unwrap()

            VIEW3D_PT_OKUnwrap.operation_time = round(
                (time.perf_counter() - start) * 1000, 2
            )
        except Exception as e:
            print(f"MESH_OT_unwrap: {e}", file=sys.stderr)
        finally:
            endUnwrapBatch(batch)
        print("====================================================")

        return {"FINISHED"}


class MESH_OT_unload_dll(bpy.types.Operator):
    """Create a new monkey mesh object with a subdivision surf modifier and shaded smooth"""

    bl_idname = "mesh.unload_dll"
    bl_label = "PLACEHOLDER"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        unloadLibrary()
        return {"FINISHED"}


class MESH_OT_load_dll(bpy.types.Operator):
    """Create a new monkey mesh object with a subdivision surf modifier and shaded smooth"""

    bl_idname = "mesh.load_dll"
    bl_label = "PLACEHOLDER"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        global okunwrap_dll
        okunwrap_dll = CDLL(getLibraryPath())

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
        row.operator("mesh.load_dll", text="LOAD DLL")
        row = self.layout.row()
        row.operator("mesh.unload_dll", text="UNLOAD DLL")

        self.layout.separator()
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "unwrapSteps", text="Unwrap Steps")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "curvatureThreshold", text="Curvature Threshold")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "extendAmount", text="Extend Amount")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "seamMarginAmount", text="Seam Margin Amount")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "seamSearchRadius", text="Seam Search Radius")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "overwriteSeams", text="Overwrite Seams")
        row.prop(CURVATURE_Properties, "unwrapAtEnd", text="Unwrap UV")
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "useSharpAsSeams")
        self.layout.separator()

        if bpy.app.version >= (4, 3, 0):
            self.layout.label(text="Unwrap Type:")
            row = self.layout.row()
            row.prop(CURVATURE_Properties, "unwrapType", text="")

        self.layout.separator()
        row = self.layout.row()
        row.prop(CURVATURE_Properties, "biasCurvatureAmount", text="Curvature Bias")
        row = self.layout.row()
        row.prop(
            CURVATURE_Properties,
            "biasInlineLoopnessAmount",
            text="Inline Loopness Bias",
        )
        row = self.layout.row()
        row.prop(
            CURVATURE_Properties, "biasSeamClosenessAmount", text="Seam Closeness Bias"
        )
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
        default='ANGLE_BASED',
        items= [('ANGLE_BASED', "Angle Based", ""),
                ('CONFORMAL', "Conformal", ""),
                ('MINIMUM_STRETCH', "Minimum Stretch", "")
        ],
        description="Unwrap type - Angle Based, Conformal or Minimum Stretch",
    )  # type: ignore

    useSharpAsSeams: bpy.props.BoolProperty(
        name="Mark Sharp Edges as Seams",
        default=True,
        description="Option will mark all sharp edges as seams before running the algorithm",
    ) # type: ignore


classes = [
    VIEW3D_PT_OKUnwrap,
    MESH_OT_unwrap,
    MESH_OT_unload_dll,
    MESH_OT_load_dll,
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
