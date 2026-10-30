"""Viewport Gaussian splat viewer: draws splat objects with a 3D Gaussian Splatting rasterizer in every 3D viewport.

The projection and spherical-harmonic colour follow the reference 3DGS renderer (Kerbl et al. 2023), as in
KIRI Engine's 3DGS Render add-on (Apache-2.0, see THIRD_PARTY.md). The data path is ours: the PLY is read from
disk in a background thread (nothing large is stored in the .blend), textures are filled straight from numpy,
and draw order is re-sorted in a background thread when the view changes, so navigation never waits on it.
A splat object is an Empty with role 'gaussian_splat' and the PLY path in 'vipe_splat_ply'; its world matrix
places the PLY's (ViPE world) coordinates, so it follows the job's alignment root.
"""
import os
import threading

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import splat_data

ROLE = 'gaussian_splat'
WIDTH = 4096  # texels per row of every data texture

VERTEX_SOURCE = """
#define SH_C0 0.28209479177387814
#define SH_C1 0.4886025119029199
vec4 base_texel(int i) { return texelFetch(base, ivec2(i % texture_width, i / texture_width), 0); }
vec4 sh_texel(int i) { return texelFetch(sh, ivec2(i % texture_width, i / texture_width), 0); }
float s[48];
vec3 coeff(int k) { return vec3(s[3 * k], s[3 * k + 1], s[3 * k + 2]); }

vec3 colour(int g, vec3 d) {
    int texels = sh_degree == 0 ? 1 : (sh_degree == 1 ? 3 : (sh_degree == 2 ? 7 : 12));
    for (int i = 0; i < texels; i++) {
        vec4 t = sh_texel(g * 12 + i);
        s[4 * i] = t.x; s[4 * i + 1] = t.y; s[4 * i + 2] = t.z; s[4 * i + 3] = t.w;
    }
    vec3 c = SH_C0 * coeff(0);
    if (sh_degree >= 1) {
        float x = d.x, y = d.y, z = d.z;
        c += -SH_C1 * y * coeff(1) + SH_C1 * z * coeff(2) - SH_C1 * x * coeff(3);
        if (sh_degree >= 2) {
            float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
            c += 1.0925484305920792 * xy * coeff(4) - 1.0925484305920792 * yz * coeff(5)
               + 0.31539156525252005 * (2.0 * zz - xx - yy) * coeff(6) - 1.0925484305920792 * xz * coeff(7)
               + 0.5462742152960396 * (xx - yy) * coeff(8);
            if (sh_degree >= 3) {
                c += -0.5900435899266435 * y * (3.0 * xx - yy) * coeff(9) + 2.890611442640554 * xy * z * coeff(10)
                   - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * coeff(11)
                   + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeff(12)
                   - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * coeff(13)
                   + 1.445305721320277 * z * (xx - yy) * coeff(14) - 0.5900435899266435 * x * (xx - 3.0 * yy) * coeff(15);
            }
        }
    }
    return max(c + 0.5, vec3(0.0));
}

void cull() { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); v_alpha = 0.0; }

void main() {
    int instance = gl_InstanceID;
    int g = int(texelFetch(draw_order, ivec2(instance % texture_width, instance / texture_width), 0).r);
    vec4 p = base_texel(g * 3);
    vec4 c0 = base_texel(g * 3 + 1);
    vec4 c1 = base_texel(g * 3 + 2);
    vec4 view = ModelViewMatrix * vec4(p.xyz, 1.0);
    vec4 clip = ProjectionMatrix * view;
    if (clip.w <= 1e-6) { cull(); return; }
    vec3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3 || ndc.z < -1.0 || ndc.z > 1.0) { cull(); return; }
    // 3D covariance in view space, then the 2D screen covariance through the projection's Jacobian (pixels).
    mat3 W = mat3(ModelViewMatrix);
    mat3 V = W * mat3(c0.x, c0.y, c0.z, c0.y, c0.w, c1.x, c0.z, c1.x, c1.y) * transpose(W);
    float fx = ProjectionMatrix[0][0] * 0.5 * viewport_size.x;
    float fy = ProjectionMatrix[1][1] * 0.5 * viewport_size.y;
    vec3 j0 = vec3(fx, 0.0, 0.0);
    vec3 j1 = vec3(0.0, fy, 0.0);
    if (ProjectionMatrix[3][3] < 0.5) {
        float z = -view.z;
        float tx = clamp(view.x / z, -1.3 / ProjectionMatrix[0][0], 1.3 / ProjectionMatrix[0][0]);
        float ty = clamp(view.y / z, -1.3 / ProjectionMatrix[1][1], 1.3 / ProjectionMatrix[1][1]);
        j0 = vec3(fx / z, 0.0, fx * tx / z);
        j1 = vec3(0.0, fy / z, fy * ty / z);
    }
    float a = dot(j0, V * j0) + 0.3;
    float b = dot(j0, V * j1);
    float c = dot(j1, V * j1) + 0.3;
    float mid = 0.5 * (a + c);
    float radius = length(vec2(0.5 * (a - c), b));
    float minor_variance = mid - radius;
    if (minor_variance <= 0.0) { cull(); return; }
    vec2 axis = abs(b) > 1e-12 ? normalize(vec2(b, mid + radius - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 major = axis * min(3.0 * sqrt(mid + radius), 1024.0);
    vec2 minor = vec2(-axis.y, axis.x) * min(3.0 * sqrt(minor_variance), 1024.0);
    gl_Position = vec4(ndc.xy + (quad.x * major + quad.y * minor) * 2.0 / viewport_size, ndc.z, 1.0);
    v_q = quad * 3.0;
    v_alpha = p.w;
    v_depth = ndc.z * 0.5 + 0.5;
    v_color = colour(g, normalize(p.xyz - camera_object));
}
"""

FRAGMENT_SOURCE = """
void main() {
    float d2 = dot(v_q, v_q);
    if (d2 > 9.0) discard;
    float alpha = min(0.99, v_alpha * exp(-0.5 * d2));
    if (alpha < 1.0 / 255.0) discard;
    if (use_depth != 0 && v_depth > texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0).r) discard;
    frag_color = vec4(v_color * alpha, alpha);  // premultiplied, blended back to front
}
"""

COMPOSITE_VERTEX = """
void main() { uv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }
"""

COMPOSITE_FRAGMENT = """
void main() {
    vec4 c = texture(image, uv);
    if (c.a <= 0.0) discard;
    // Splat colours are display-referred, like the plate they were trained on; the viewport target is linear (sRGB-encoded).
    vec3 s = c.rgb / c.a;
    vec3 linear = mix(s / 12.92, pow((s + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), s));
    frag_color = vec4(linear * c.a, c.a);
}
"""

_shaders = None
_splats = {}  # PLY path -> _Splat
_targets = {}  # viewport size -> (texture, framebuffer)
_handle = _text_handle = None
_no_depth = None


def _create_shaders():
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC2', 'quad')
    for kind, name in [('MAT4', 'ModelViewMatrix'), ('MAT4', 'ProjectionMatrix'), ('VEC2', 'viewport_size'), ('VEC3', 'camera_object'),
                       ('INT', 'sh_degree'), ('INT', 'texture_width'), ('INT', 'use_depth')]:
        info.push_constant(kind, name)
    for slot, name in enumerate(['base', 'sh', 'draw_order', 'scene_depth']):
        info.sampler(slot, 'FLOAT_2D', name)
    interface = gpu.types.GPUStageInterfaceInfo('vipe_splat_interface')
    interface.smooth('VEC3', 'v_color')
    interface.smooth('FLOAT', 'v_alpha')
    interface.smooth('VEC2', 'v_q')
    interface.smooth('FLOAT', 'v_depth')
    info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', 'frag_color')
    info.vertex_source(VERTEX_SOURCE)
    info.fragment_source(FRAGMENT_SOURCE)
    splat = gpu.shader.create_from_info(info)
    quad = batch_for_shader(splat, 'TRIS', {'quad': [(-1, -1), (1, -1), (1, 1), (-1, 1)]}, indices=[(0, 1, 2), (0, 2, 3)])
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, 'VEC2', 'position')
    composite_interface = gpu.types.GPUStageInterfaceInfo('vipe_splat_composite_interface')
    composite_interface.smooth('VEC2', 'uv')
    info.vertex_out(composite_interface)
    info.sampler(0, 'FLOAT_2D', 'image')
    info.fragment_out(0, 'VEC4', 'frag_color')
    info.vertex_source(COMPOSITE_VERTEX)
    info.fragment_source(COMPOSITE_FRAGMENT)
    composite = gpu.shader.create_from_info(info)
    screen = batch_for_shader(composite, 'TRI_FAN', {'position': [(-1, -1), (1, -1), (1, 1), (-1, 1)]})
    return splat, quad, composite, screen


def _texture(array, texel_format):
    """A WIDTH-wide data texture from a flat float32 array whose length is a whole number of rows."""
    rows = len(array) // (WIDTH * (1 if texel_format == 'R32F' else 4))
    return gpu.types.GPUTexture((WIDTH, rows), format=texel_format, data=gpu.types.Buffer('FLOAT', len(array), array))


def _padded(order):
    out = np.zeros(-(-len(order) // WIDTH) * WIDTH, np.float32)
    out[:len(order)] = order
    return out


class _Order:
    """Back-to-front draw order for one viewport, re-sorted off the main thread."""
    def __init__(self):
        self.texture = self.row = self.thread = self.result = None


class _Splat:
    def __init__(self, path):
        self.path, self.mtime = path, os.path.getmtime(path)
        self.data = self.textures = self.error = None
        self.percent = self.ids = self.points = None
        self.generation, self.loaded = 0, 0.0
        self.orders = {}
        self.loader = threading.Thread(target=self._load, daemon=True)
        self.loader.start()

    def _load(self):
        try:
            self.data = splat_data.load_gaussians(self.path, pad=WIDTH, progress=lambda fraction: setattr(self, 'loaded', fraction))
        except Exception as exc:  # shown in the panel
            self.error = str(exc)

    @property
    def busy(self):
        return self.error is None and (self.textures is None or any(o.thread is not None for o in self.orders.values()))

    def upload(self):
        """GPU textures from the loaded arrays (main thread); the large arrays are dropped afterwards."""
        data = self.data
        self.textures = (_texture(data['base'].reshape(-1), 'RGBA32F'), _texture(data['sh'].reshape(-1), 'RGBA16F'))
        self.sh_degree = data['sh_degree']
        self.data = dict(positions=data['positions'], importance=data['importance'], count=data['count'])

    def select(self, percent):
        if percent != self.percent:
            self.percent, self.generation = percent, self.generation + 1
            self.ids = splat_data.display_subset(self.data['importance'], percent)
            self.points = self.data['positions'][self.ids]
            self.orders.clear()

    def order(self, key, row):
        """This viewport's draw-order texture: sorted now the first time, afterwards in the background when the view moved."""
        order = self.orders.setdefault(key, _Order())
        result = order.result
        if result is not None and result[0] == self.generation:
            order.texture, order.row = _texture(result[2], 'R32F'), result[1]
        if result is not None:
            order.result = order.thread = None
        if order.texture is None:
            order.texture, order.row = _texture(_padded(splat_data.back_to_front(self.points, self.ids, row)), 'R32F'), row
        elif order.thread is None and max(abs(a - b) for a, b in zip(row, order.row)) > 1e-4:
            generation, points, ids = self.generation, self.points, self.ids

            def sort():
                order.result = (generation, row, _padded(splat_data.back_to_front(points, ids, row)))
            order.thread = threading.Thread(target=sort, daemon=True)
            order.thread.start()
        return order.texture


def status(path):
    """'' when the splat is drawing, else a short state for the panel."""
    splat = _splats.get(str(path))
    if splat is None:
        return ''
    if splat.error:
        return 'Could not load the splat: ' + splat.error
    if splat.textures is None:
        return f'Loading Gaussian splat… {splat.loaded:.0%}'
    return ''


def gaussian_count(path):
    splat = _splats.get(str(path))
    return splat.data['count'] if splat and splat.data else 0


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _poll():
    redraw()
    return 0.05 if any(s.busy for s in _splats.values()) else None


def _splat_for(path):
    splat = _splats.get(path)
    try:
        if splat is None or (splat.textures is not None and os.path.getmtime(path) != splat.mtime):
            splat = _splats[path] = _Splat(path)
    except OSError:
        return None
    if splat.textures is None and splat.data is not None:
        splat.upload()
    return splat if splat.textures is not None else None


def _target(width, height):
    if (width, height) not in _targets:
        _targets.clear()
        texture = gpu.types.GPUTexture((width, height), format='RGBA16F')
        _targets[(width, height)] = texture, gpu.types.GPUFrameBuffer(color_slots=[texture])
    return _targets[(width, height)]


def _draw():
    global _shaders, _no_depth
    context = bpy.context
    wanted = {o.get('vipe_splat_ply') for o in bpy.data.objects if o.get('vipe_role') == ROLE}
    for stale in [path for path in _splats if path not in wanted]:
        del _splats[stale]  # e.g. a retrained splat replaced it; hidden viewers keep theirs for an instant Show
    objects = [o for o in context.view_layer.objects if o.get('vipe_role') == ROLE and o.get('vipe_splat_ply') and o.visible_get()]
    if not objects:
        return
    try:
        settings = context.scene.vipe_remote
        splats = [(o, _splat_for(o['vipe_splat_ply'])) for o in objects]
        if any(s is None or s.busy for _, s in splats) and not bpy.app.timers.is_registered(_poll):
            bpy.app.timers.register(_poll, first_interval=0.05)
        splats = [(o, s) for o, s in splats if s is not None]
        if not splats:
            return
        if _shaders is None:
            _shaders = _create_shaders()
        shader, quad, composite, screen = _shaders
        x, y, width, height = gpu.state.viewport_get()
        view, projection = gpu.matrix.get_model_view_matrix().copy(), gpu.matrix.get_projection_matrix().copy()
        if settings.splat_occlusion:
            depth = gpu.state.active_framebuffer_get().read_depth(0, 0, width, height)
            depth.dimensions = width * height
            depth = gpu.types.GPUTexture((width, height), format='R32F', data=depth)
        elif _no_depth is None:
            _no_depth = gpu.types.GPUTexture((1, 1), format='R32F', data=gpu.types.Buffer('FLOAT', 1, [1.0]))
        texture, framebuffer = _target(width, height)
        region = context.region.as_pointer() if context.region else 0
        with framebuffer.bind():
            framebuffer.clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set('ALPHA_PREMULT')
            shader.bind()
            shader.uniform_float('ProjectionMatrix', projection)
            shader.uniform_float('viewport_size', (width, height))
            shader.uniform_int('texture_width', WIDTH)
            shader.uniform_int('use_depth', int(settings.splat_occlusion))
            shader.uniform_sampler('scene_depth', depth if settings.splat_occlusion else _no_depth)
            for obj, splat in splats:
                splat.select(settings.splat_display_percent)
                model_view = view @ obj.matrix_world
                order = splat.order(region, tuple(model_view[2]))
                shader.uniform_float('ModelViewMatrix', model_view)
                shader.uniform_float('camera_object', model_view.inverted_safe().translation)
                shader.uniform_int('sh_degree', min(settings.splat_sh_degree, splat.sh_degree))
                shader.uniform_sampler('base', splat.textures[0])
                shader.uniform_sampler('sh', splat.textures[1])
                shader.uniform_sampler('draw_order', order)
                quad.draw_instanced(shader, instance_count=len(splat.ids))
        gpu.state.blend_set('ALPHA_PREMULT')
        composite.bind()
        composite.uniform_sampler('image', texture)
        screen.draw(composite)
        gpu.state.blend_set('NONE')
        if any(s.busy for _, s in splats) and not bpy.app.timers.is_registered(_poll):
            bpy.app.timers.register(_poll, first_interval=0.05)
    except Exception as exc:  # never break the viewport; report once per message
        message = f'Gaussian splat viewer: {exc}'
        if message != getattr(_draw, 'last_error', None):
            _draw.last_error = message
            print(message)


def clear():
    """Free all GPU and CPU splat data (new file, add-on disabled)."""
    _splats.clear()
    _targets.clear()


def _draw_status():
    """Loading progress or a load error in the viewport, so a splat that is not drawn yet is never silent."""
    context = bpy.context
    lines = []
    for obj in context.view_layer.objects:
        if obj.get('vipe_role') == ROLE and obj.get('vipe_splat_ply') and obj.visible_get():
            splat = _splats.get(obj['vipe_splat_ply'])
            if not os.path.isfile(obj['vipe_splat_ply']):
                text = 'Could not load the splat: file not found ' + obj['vipe_splat_ply']
            else:
                text = status(obj['vipe_splat_ply']) if splat else 'Loading Gaussian splat…'
            if text:
                lines.append(f'{obj.name}: {text}')
    if not lines:
        return
    scale = context.preferences.view.ui_scale
    blf.size(0, 13 * scale)
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 5, 0.0, 0.0, 0.0, 1.0)
    for i, line in enumerate(lines):
        error = 'Could not load' in line
        blf.color(0, 1.0, 0.45 if error else 1.0, 0.35 if error else 1.0, 1.0)
        blf.position(0, 20 * scale, (60 + 20 * i) * scale, 0)
        blf.draw(0, line)
    blf.disable(0, blf.SHADOW)


def register():
    global _handle, _text_handle
    if _handle is None and not bpy.app.background:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
        _text_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_status, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _handle, _text_handle, _shaders, _no_depth
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        bpy.types.SpaceView3D.draw_handler_remove(_text_handle, 'WINDOW')
        _handle = _text_handle = None
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    clear()
    _shaders = _no_depth = None
