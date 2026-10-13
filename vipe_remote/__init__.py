bl_info = {
    "name": "ViPE Remote Tracking",
    "author": "Camera Tracking R&D",
    "version": (0, 4, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > ViPE Remote",
    "description": "Submit a video to ComfyUI and import its solved camera and native 3D points",
    "category": "Camera",
}


def register():
    from . import addon
    addon.register()


def unregister():
    from . import addon
    addon.unregister()
