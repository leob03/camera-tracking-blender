bl_info = {
    "name": "Camera Track",
    "author": "Camera Tracking R&D",
    "version": (0, 6, 1),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Camera Track",
    "description": "Submit a movie or image-sequence plate to ComfyUI, import the solved camera and points, export to Nuke",
    "category": "Camera",
}


def register():
    from . import addon
    addon.register()


def unregister():
    from . import addon
    addon.unregister()
