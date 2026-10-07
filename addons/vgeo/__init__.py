"""VGEO - virtualized geometry for Blender.

Converts a heavy mesh into a cluster-LOD DAG (.vgeo) and streams a
view-dependent cut back into ordinary meshes, so EEVEE and Cycles render it
with full materials while Blender only ever holds what the view needs.
"""

bl_info = {
    "name": "VGEO Virtualized Geometry",
    "author": "Lucas DeMeritt",
    "version": (0, 2, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > VGEO",
    "description": "Nanite-style streamed LOD for huge meshes, rendered natively by EEVEE and Cycles",
    "category": "Object",
}

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import build, instances, native, stream


def _invalidate(self, _context):
    rt = stream._runtimes.get(self.uid)
    if rt:
        rt.invalidate()


class VGEOObjectSettings(bpy.types.PropertyGroup):
    uid: StringProperty(name="Asset ID", options={'HIDDEN'})
    path: StringProperty(name="File", subtype='FILE_PATH', update=_invalidate,
                         options={'PATH_SUPPORTS_BLEND_RELATIVE'})
    collection: PointerProperty(type=bpy.types.Collection, name="Chunks")
    source: PointerProperty(type=bpy.types.Object, name="Source")
    source_triangles: IntProperty(name="Source Triangles", options={'HIDDEN'})
    file_bytes: IntProperty(name="File Size", options={'HIDDEN'})
    pixel_error: FloatProperty(name="Viewport Error", default=1.0, min=0.1, soft_max=8.0,
                               subtype='PIXEL', update=_invalidate,
                               description="Largest allowed geometric error on screen, in pixels (viewport)")
    render_pixel_error: FloatProperty(name="Render Error", default=0.5, min=0.05, soft_max=4.0,
                                      subtype='PIXEL',
                                      description="Largest allowed geometric error in the final render, in pixels")
    offscreen: EnumProperty(
        name="Off-screen", default='COARSEN', update=_invalidate,
        items=[('COARSEN', "Coarsen", "Keep geometry outside the view at reduced detail, so it still casts "
                                      "shadows and shows in reflections"),
               ('CULL', "Cull", "Drop geometry outside the viewport (fastest; final renders coarsen instead)"),
               ('FULL', "Full", "Treat off-screen geometry like on-screen geometry")])
    offscreen_scale: FloatProperty(name="Off-screen Error", default=8.0, min=1.0, soft_max=64.0,
                                   update=_invalidate,
                                   description="Error multiplier for geometry outside the view")
    freeze: BoolProperty(name="Freeze", default=False,
                         description="Stop updating the cut as the view moves")
    lod_colors: BoolProperty(name="LOD Colors", default=False, update=_invalidate,
                             description="Write a 'vgeo_lod' color attribute (view it with Solid shading, "
                                         "Color: Attribute)")
    # set on the streamed copies an instancer manages (see instances.py): owning instancer's uid,
    # and the placement this copy currently shows (-1 = none)
    slot_of: StringProperty(options={'HIDDEN'})
    slot_index: IntProperty(default=-1, options={'HIDDEN'})


class VGEOInstanceSettings(bpy.types.PropertyGroup):
    uid: StringProperty(name="Instancer ID", options={'HIDDEN'})
    source: PointerProperty(type=bpy.types.Object, name="Asset",
                            description="The virtualized object these placements show")
    levels: PointerProperty(type=bpy.types.Collection, name="Levels")
    pixel_error: FloatProperty(name="Viewport Error", default=1.0, min=0.1, soft_max=8.0, subtype='PIXEL')
    render_pixel_error: FloatProperty(name="Render Error", default=0.5, min=0.05, soft_max=4.0, subtype='PIXEL')
    min_level: IntProperty(name="Finest Level", default=0, min=0, soft_max=8,
                           description="Never use levels finer than this (caps memory for very dense assets)")
    stream_slots: IntProperty(name="Streamed Copies", default=4, min=0, max=16,
                              description="Copies close enough to want full detail get a view-dependent "
                                          "streamed cut of their own instead of a whole-asset level "
                                          "(fine where you look, coarse elsewhere); at most this many")
    freeze: BoolProperty(name="Freeze", default=False)
    shown_triangles: IntProperty(name="Shown Triangles", options={'HIDDEN'})


class VGEO_OT_scatter(bpy.types.Operator):
    """Scatter copies of the active VGEO object over the other selected mesh; each copy picks its own LOD"""
    bl_idname = "vgeo.scatter"
    bl_label = "Scatter Instances"
    bl_options = {'REGISTER', 'UNDO'}

    count: IntProperty(name="Count", default=500, min=1, max=1_000_000)
    seed: IntProperty(name="Seed", default=0)
    scale_min: FloatProperty(name="Scale Min", default=0.7, min=0.001)
    scale_max: FloatProperty(name="Scale Max", default=1.3, min=0.001)
    align: BoolProperty(name="Align to Surface", default=True)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None and ob.vgeo.uid and
                any(o is not ob and o.type == 'MESH' for o in context.selected_objects))

    def execute(self, context):
        source = context.active_object
        surface = next(o for o in context.selected_objects if o is not source and o.type == 'MESH')
        try:
            inst = instances.create_instancer(context, source, surface, self.count, self.seed,
                                              (self.scale_min, max(self.scale_min, self.scale_max)), self.align)
        except (ValueError, RuntimeError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        views = stream.viewport_views() if context.window_manager.windows else []
        if views:
            instances.update(inst, views, inst.vgeo_inst.pixel_error)
        for o in context.view_layer.objects:
            o.select_set(False)
        inst.select_set(True)
        context.view_layer.objects.active = inst
        self.report({'INFO'}, f"Scattered {self.count} instances of {source.name}")
        return {'FINISHED'}


class VGEO_OT_rebuild_levels(bpy.types.Operator):
    """Rebuild the instancer's level meshes from its asset (after re-virtualizing or changing materials)"""
    bl_idname = "vgeo.rebuild_levels"
    bl_label = "Rebuild Levels"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.vgeo_inst.uid and ob.vgeo_inst.source is not None

    def execute(self, context):
        ob = context.active_object
        instances.build_levels(ob, ob.vgeo_inst.source)
        return {'FINISHED'}


class VGEO_OT_virtualize(bpy.types.Operator):
    """Build a streamed LOD hierarchy from the active mesh and replace it with a VGEO proxy"""
    bl_idname = "vgeo.virtualize"
    bl_label = "Virtualize Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    remove_source: BoolProperty(name="Remove Source Mesh", default=False,
                                description="Delete the original object to free its memory "
                                            "(otherwise it is hidden and kept)")
    max_triangles: IntProperty(name="Cluster Size", default=128, min=32, max=256,
                               description="Triangles per cluster")
    target_chunks: IntProperty(name="Chunks", default=0, min=0, max=8192,
                               description="Streaming regions (0 = automatic). More chunks means smaller "
                                           "updates but more objects")

    _job = None
    _timer = None

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH' and not ob.vgeo.uid and context.mode == 'OBJECT'

    def _prepare(self, context):
        src = context.active_object
        if not native.available():
            try:
                native.lib()
            except RuntimeError as e:
                self.report({'ERROR'}, str(e))
            return None
        try:
            arrays = build.mesh_arrays(src, context.evaluated_depsgraph_get())
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return None
        self._src_name = src.name
        self._uid = stream.new_uid()
        self._full, self._setting = build.default_path(src, self._uid)
        self._materials = [m.name if m else None for m in arrays["material_list"]]
        names = [n or "" for n in self._materials]
        params = build.material_params(arrays["material_list"])
        return build.Job(arrays, self._full, names, self.max_triangles, self.target_chunks, params)

    def _finish(self, context, job):
        src = bpy.data.objects.get(self._src_name)
        if job.error is not None:
            if isinstance(job.error, InterruptedError):
                self.report({'WARNING'}, "Virtualize cancelled")
                return {'CANCELLED'}
            self.report({'ERROR'}, f"VGEO build failed: {job.error}")
            return {'CANCELLED'}
        if src is None:
            self.report({'ERROR'}, "Source object disappeared during the build")
            return {'CANCELLED'}
        s = job.result
        materials = [bpy.data.materials.get(n) if n else None for n in self._materials]
        build.create_proxy(context, src, self._setting, self._uid, s, self.remove_source, materials)
        self.report({'INFO'}, f"VGEO: {s['source_triangles']:,} triangles -> {s['clusters']:,} clusters, "
                              f"{s['lod_levels']} levels in {s['seconds']:.1f}s")
        return {'FINISHED'}

    def execute(self, context):
        job = self._prepare(context)
        if job is None:
            return {'CANCELLED'}
        job._run()
        return self._finish(context, job)

    def invoke(self, context, event):
        job = self._prepare(context)
        if job is None:
            return {'CANCELLED'}
        self._job = job
        job.thread.start()
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        job = self._job
        if event.type == 'ESC':
            job.cancel = True
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'} if job.thread.is_alive() else self._end(context)
        context.window_manager.progress_update(int(job.overall * 100))
        stage = ("Welding", "Building LOD hierarchy", "Writing")[min(job.stage, 2)]
        context.workspace.status_text_set(f"VGEO: {stage} {int(job.overall * 100)}%  (Esc to cancel)")
        if job.thread.is_alive():
            return {'RUNNING_MODAL'}
        return self._end(context)

    def _end(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        context.workspace.status_text_set(None)
        return self._finish(context, self._job)


class VGEO_OT_restore(bpy.types.Operator):
    """Remove the VGEO proxy and bring back the original mesh"""
    bl_idname = "vgeo.restore"
    bl_label = "Restore Source"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.vgeo.uid and ob.vgeo.source is not None

    def execute(self, context):
        proxy = context.active_object
        src = proxy.vgeo.source
        uid = proxy.vgeo.uid
        col = proxy.vgeo.collection
        rt = stream._runtimes.pop(uid, None)
        if rt:
            rt.close()
        if col:
            for ob in list(col.objects):
                bpy.data.objects.remove(ob)
            bpy.data.collections.remove(col)
        prefix = f"vgeo.{uid}."
        for me in [m for m in bpy.data.meshes if m.name.startswith(prefix) and m.users == 0]:
            bpy.data.meshes.remove(me)
        pm = proxy.data
        bpy.data.objects.remove(proxy)
        if pm.users == 0:
            bpy.data.meshes.remove(pm)
        src.hide_render = False
        src.hide_set(False)
        src.select_set(True)
        context.view_layer.objects.active = src
        return {'FINISHED'}


class VGEO_OT_export_web(bpy.types.Operator):
    """Write a ready-to-host web folder: compact .vgeow asset, WebGPU viewer and a page"""
    bl_idname = "vgeo.export_web"
    bl_label = "Export for Web"

    directory: StringProperty(subtype='DIR_PATH')
    with_page: BoolProperty(name="Include Viewer Page", default=True,
                            description="Also write the viewer module, decoder and an index.html")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.vgeo.uid

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from . import webexport
        try:
            out = webexport.export(context.active_object, bpy.path.abspath(self.directory), self.with_page)
        except (RuntimeError, OSError) as e:
            self.report({'ERROR'}, f"Web export failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Web export: {out['asset']} ({out['bytes'] / 2**20:.1f} MB)")
        return {'FINISHED'}


class VGEO_OT_refresh(bpy.types.Operator):
    """Reopen the .vgeo file and rebuild every chunk"""
    bl_idname = "vgeo.refresh"
    bl_label = "Reload"

    def execute(self, context):
        stream.invalidate_all(close=True)
        return {'FINISHED'}


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


class VGEO_PT_panel(bpy.types.Panel):
    bl_label = "VGEO"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VGEO"

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        if not native.available():
            layout.label(text="Native library missing", icon='ERROR')
            layout.label(text=native.library_path())
            return
        from . import livedraw
        row = layout.row(align=True)
        row.prop(context.scene, "vgeo_live_draw", toggle=True, icon='SHADING_TEXTURE')
        if context.scene.vgeo_live_draw:
            if livedraw.is_active():
                layout.label(text=f"Drawing {livedraw.drawn_triangles():,} triangles", icon='INFO')
            else:
                layout.label(text="Off while a view is in Rendered shading", icon='INFO')
        if ob is not None and ob.vgeo_inst.uid:
            vi = ob.vgeo_inst
            box = layout.box()
            col = box.column(align=True)
            col.label(text=f"Instances of {vi.source.name if vi.source else '?'}", icon='OUTLINER_OB_POINTCLOUD')
            col.label(text=f"Placements: {len(ob.data.vertices):,}")
            col.label(text=f"Showing: {vi.shown_triangles:,} triangles")
            tris = ob.get("vgeo_level_tris")
            if tris:
                col.label(text=f"All at full detail: {len(ob.data.vertices) * tris[0]:,}")
            layout.prop(vi, "pixel_error")
            layout.prop(vi, "render_pixel_error")
            layout.prop(vi, "min_level")
            layout.prop(vi, "stream_slots")
            row = layout.row(align=True)
            row.prop(vi, "freeze", toggle=True, icon='FREEZE')
            row.operator(VGEO_OT_rebuild_levels.bl_idname, icon='FILE_REFRESH')
            return
        if ob is None or not ob.vgeo.uid:
            layout.operator(VGEO_OT_virtualize.bl_idname, icon='MOD_DECIM')
            if ob is not None and ob.type == 'MESH':
                layout.label(text=f"{len(ob.data.polygons):,} faces")
            return
        v = ob.vgeo
        rt = stream._runtimes.get(v.uid)
        box = layout.box()
        if rt is None or rt.asset is None:
            box.label(text=(rt.error if rt else "Waiting for first update"), icon='INFO')
        else:
            src = rt.asset.info["source_triangles"]
            col = box.column(align=True)
            col.label(text=f"Source: {src:,} triangles")
            share = (100.0 * rt.triangles / src) if src else 0
            col.label(text=f"Showing: {rt.triangles:,} ({share:.1f}%)")
            col.label(text=f"Clusters: {rt.clusters:,} / {rt.asset.info['cluster_count']:,}")
            col.label(text=f"Levels: {rt.asset.info['lod_levels']}   Chunks: {rt.asset.chunk_count}")
            col.label(text=f"Last update: {rt.last_ms:.0f} ms, {rt.last_rebuilt} chunks")
            col.label(text=f"File: {_fmt_bytes(v.file_bytes)}")
        col = layout.column()
        col.prop(v, "pixel_error")
        col.prop(v, "render_pixel_error")
        row = layout.row(align=True)
        row.prop(v, "offscreen", expand=True)
        if v.offscreen == 'COARSEN':
            layout.prop(v, "offscreen_scale")
        row = layout.row(align=True)
        row.prop(v, "freeze", toggle=True, icon='FREEZE')
        row.prop(v, "lod_colors", toggle=True, icon='COLOR')
        layout.prop(v, "path")
        row = layout.row(align=True)
        row.operator(VGEO_OT_refresh.bl_idname, icon='FILE_REFRESH')
        row.operator(VGEO_OT_restore.bl_idname, icon='LOOP_BACK')
        layout.operator(VGEO_OT_export_web.bl_idname, icon='WORLD')
        layout.operator(VGEO_OT_scatter.bl_idname, icon='OUTLINER_OB_POINTCLOUD')


classes = (VGEOObjectSettings, VGEOInstanceSettings, VGEO_OT_virtualize, VGEO_OT_restore, VGEO_OT_export_web,
           VGEO_OT_scatter, VGEO_OT_rebuild_levels, VGEO_OT_refresh, VGEO_PT_panel)


def _menu(self, _context):
    self.layout.operator(VGEO_OT_virtualize.bl_idname, icon='MOD_DECIM')


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Object.vgeo = PointerProperty(type=VGEOObjectSettings)
    bpy.types.Object.vgeo_inst = PointerProperty(type=VGEOInstanceSettings)
    bpy.types.Scene.vgeo_live_draw = BoolProperty(
        name="Live Draw", default=True,
        description="Draw virtualized assets straight from GPU buffers in Solid and Material Preview views "
                    "(fast, view-dependent cuts for every copy). Final renders and Rendered views use EEVEE "
                    "or Cycles with real meshes either way")
    bpy.types.VIEW3D_MT_object.append(_menu)
    stream.register()
    from . import livedraw
    livedraw.register()


def unregister():
    from . import livedraw
    livedraw.unregister()
    stream.unregister()
    bpy.types.VIEW3D_MT_object.remove(_menu)
    del bpy.types.Scene.vgeo_live_draw
    del bpy.types.Object.vgeo_inst
    del bpy.types.Object.vgeo
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
