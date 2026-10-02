"""
pairwise-shader-system - ShaderSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class ShaderSystem:
    def __init__(self):
        self.state = {}

    def compile_shader(self, *args, **kwargs):
        """BUG: Shader编译不做预处理，宏定义不生效"""
        return None

    def create_material_instance(self, *args, **kwargs):
        """BUG: 材质实例不继承父材质参数，改了父材质实例不变"""
        return None

    def calculate_pbr(self, *args, **kwargs):
        """BUG: PBR计算只做漫反射，不做法线和粗糙度"""
        return None

    def compile_variant(self, *args, **kwargs):
        """BUG: Shader变体不做组合管理，10个宏就是1024个变体全编译"""
        return None

    def sort_materials(self, *args, **kwargs):
        """BUG: 材质排序只按Shader类型，不按透明度和距离"""
        return None

