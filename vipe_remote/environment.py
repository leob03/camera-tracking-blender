"""Build separately switchable, packed dense surfaces and colored depth samples."""
from pathlib import Path
import tempfile
import bpy
import numpy as np


def _mesh_object(collection, root, name, points, faces):
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(points.tolist(), [], faces.tolist())
    mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    collection.objects.link(obj)
    obj.parent=root
    return obj


def import_environment(parent, root, items):
    group=bpy.data.collections.new('05 Dense environment')
    parent.children.link(group)
    group['vipe_role']='dense_environment_group'
    group['description']='Frozen-camera depth samples; unmerged surfaces, not a watertight model.'
    samples=bpy.data.collections.new('Colored depth samples')
    surfaces=bpy.data.collections.new('Textured per-view surfaces')
    group.children.link(samples);group.children.link(surfaces)
    for item in items:
        view,data=item['view'],item['arrays']
        index=view['source_frame_index']+1
        obj=_mesh_object(surfaces,root,f'Depth surface source {index}',data['points'],data['faces'])
        obj['vipe_role']='dense_surface'
        obj['source_frame']=index
        uv=obj.data.uv_layers.new(name='Plate UV')
        loops=np.empty(len(obj.data.loops),dtype=np.int32)
        obj.data.loops.foreach_get('vertex_index',loops)
        uv.data.foreach_set('uv',data['uv'][loops].reshape(-1))
        with tempfile.TemporaryDirectory(prefix='vipe-texture-') as temp:
            path=Path(temp)/f'plate_{index}.png'
            path.write_bytes(item['texture'])
            image=bpy.data.images.load(str(path),check_existing=False)
            image.pack()
            image.filepath=f'//vipe_packed/plate_{index}.png'
        material=bpy.data.materials.new(f'Plate source {index}')
        material.use_nodes=True
        nodes=material.node_tree.nodes;nodes.clear()
        output=nodes.new('ShaderNodeOutputMaterial')
        emission=nodes.new('ShaderNodeEmission')
        texture=nodes.new('ShaderNodeTexImage');texture.image=image
        material.node_tree.links.new(texture.outputs['Color'],emission.inputs['Color'])
        material.node_tree.links.new(emission.outputs[0],output.inputs['Surface'])
        obj.data.materials.append(material)
        obj.hide_set(True)
        obj.hide_render=True
        # Small camera-facing quads reveal the accumulation without bridging holes.
        pose=np.asarray(view['c2w'])
        k=np.asarray(view['intrinsics'])
        z=data['depths']
        rx=(view['stride']*.42*z/k[0])[:,None]*pose[:3,0]
        ry=(view['stride']*.42*z/k[1])[:,None]*pose[:3,1]
        points=data['points']
        verts=np.stack((points-rx-ry,points+rx-ry,points+rx+ry,points-rx+ry),axis=1).reshape(-1,3)
        faces=np.arange(len(verts),dtype=np.int32).reshape(-1,4)
        splats=_mesh_object(samples,root,f'Colored samples source {index}',verts,faces)
        splats['vipe_role']='dense_samples';splats['source_frame']=index
        colors=np.repeat(data['colors'],4,axis=0)
        colors=np.where(colors<=.04045,colors/12.92,((colors+.055)/1.055)**2.4)
        rgba=np.concatenate((colors,np.ones((len(colors),1))),axis=1).astype(np.float32)
        attr=splats.data.color_attributes.new(name='Color',type='FLOAT_COLOR',domain='POINT')
        attr.data.foreach_set('color',rgba.reshape(-1))
        mat=bpy.data.materials.new(f'Depth colors source {index}');mat.use_nodes=True
        nodes=mat.node_tree.nodes;nodes.clear()
        output=nodes.new('ShaderNodeOutputMaterial');emission=nodes.new('ShaderNodeEmission');color=nodes.new('ShaderNodeVertexColor');color.layer_name='Color'
        mat.node_tree.links.new(color.outputs['Color'],emission.inputs['Color']);mat.node_tree.links.new(emission.outputs[0],output.inputs['Surface'])
        splats.data.materials.append(mat)
        splats.hide_render=True
    return group
