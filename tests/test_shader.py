import pytest
from shader.system import ShaderSystem


class TestShaderSystem:

    def test_compile_shader(self):
        """Shader编译不做预处理，宏定义不生效"""
        obj = ShaderSystem()
        result = obj.compile_shader()
        self.assertIsNotNone(result)

    def test_create_material_instance(self):
        """材质实例不继承父材质参数，改了父材质实例不变"""
        obj = ShaderSystem()
        result = obj.create_material_instance()
        self.assertIsNotNone(result)

    def test_calculate_pbr(self):
        """PBR计算只做漫反射，不做法线和粗糙度"""
        obj = ShaderSystem()
        result = obj.calculate_pbr()
        self.assertIsNotNone(result)

    def test_compile_variant(self):
        """Shader变体不做组合管理，10个宏就是1024个变体全编译"""
        obj = ShaderSystem()
        result = obj.compile_variant()
        self.assertIsNotNone(result)

    def test_sort_materials(self):
        """材质排序只按Shader类型，不按透明度和距离"""
        obj = ShaderSystem()
        result = obj.sort_materials()
        self.assertIsNotNone(result)

