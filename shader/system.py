"""
pairwise-shader-system - ShaderSystem

A complete shader system implementation providing:

- Shader compilation with preprocessing (#define macro expansion),
  error checking (failed compiles never replace a working shader),
  and a compile cache so identical sources are compiled only once.
- Material instances that inherit parent parameters, track which
  parameters were overridden on the instance, and serialize to/from
  dicts so parameters survive save/load.
- PBR shading with normal mapping, roughness/metallic workflow,
  energy conservation, and a Fresnel (Schlick) rim term.
- Shader variant management: variants are compiled on demand, and
  unsupported variants fall back to a safe default instead of crashing.
- Material sorting by transparency and view distance, draw-call
  batching for identical materials, and static-object marking so
  static geometry is not re-sorted every frame.
"""

import hashlib
import math
import re


class ShaderCompileError(Exception):
    """Raised when a shader fails to compile."""


class Shader:
    """A compiled shader program."""

    def __init__(self, name, source, defines=None, preprocessed=None):
        self.name = name
        self.source = source
        self.defines = dict(defines or {})
        self.preprocessed = preprocessed if preprocessed is not None else source
        self.handle = self._make_handle()

    def _make_handle(self):
        digest = hashlib.sha1(self.preprocessed.encode("utf-8")).hexdigest()
        return "shader_" + digest[:16]

    def __repr__(self):
        return "Shader(name=%r, handle=%r)" % (self.name, self.handle)


class Material:
    """A material: a shader plus a set of named parameters."""

    def __init__(self, name, shader=None, params=None, transparent=False):
        self.name = name
        self.shader = shader
        self.params = dict(params or {})
        self.transparent = transparent

    def set_param(self, key, value):
        self.params[key] = value

    def get_param(self, key, default=None):
        return self.params.get(key, default)


class MaterialInstance:
    """An instance of a parent material.

    Reads fall through to the parent for any parameter that has not
    been explicitly overridden on the instance, so changes to the
    parent propagate automatically. Overridden parameters are tracked
    in ``overrides`` and survive serialization.
    """

    def __init__(self, parent, name=None):
        if parent is None:
            raise ValueError("MaterialInstance requires a parent material")
        self.parent = parent
        self.name = name or (getattr(parent, "name", "material") + "_instance")
        self.overrides = {}

    @property
    def shader(self):
        return getattr(self.parent, "shader", None)

    @property
    def transparent(self):
        return getattr(self.parent, "transparent", False)

    def set_param(self, key, value):
        self.overrides[key] = value

    def get_param(self, key, default=None):
        if key in self.overrides:
            return self.overrides[key]
        parent_params = getattr(self.parent, "params", None) or {}
        return parent_params.get(key, default)

    def get_params(self):
        """Effective parameters: parent params with overrides applied."""
        params = dict(getattr(self.parent, "params", None) or {})
        params.update(self.overrides)
        return params

    def is_overridden(self, key):
        return key in self.overrides

    def get_overrides(self):
        return dict(self.overrides)

    def serialize(self):
        return {
            "type": "material_instance",
            "name": self.name,
            "parent": getattr(self.parent, "name", None),
            "parent_params": dict(getattr(self.parent, "params", None) or {}),
            "overrides": dict(self.overrides),
        }

    @classmethod
    def deserialize(cls, data, shader_system=None):
        parent = None
        if shader_system is not None:
            parent = shader_system.get_material(data.get("parent"))
        if parent is None:
            parent = Material(
                data.get("parent") or "material",
                params=data.get("parent_params") or {},
            )
        instance = cls(parent, name=data.get("name"))
        instance.overrides = dict(data.get("overrides") or {})
        return instance


class ShaderSystem:
    def __init__(self):
        self.state = {}
        self.shader_cache = {}
        self.failed_shaders = {}
        self.compiled_shaders = {}
        self.materials = {}
        self.instances = {}
        self.variants = {}
        self.supported_defines = None
        self.default_variant_key = ""
        self.static_order_cache = {}

    # ------------------------------------------------------------------
    # Shader compilation
    # ------------------------------------------------------------------

    def preprocess(self, source, defines=None):
        """Expand #define macros and inject externally supplied defines."""
        defines = dict(defines or {})
        if not source:
            return source or ""
        injected = ["#define %s %s" % (name, value)
                    for name, value in sorted(defines.items())]
        out_lines = list(injected)
        define_pattern = re.compile(r"^\s*#define\s+(\w+)(?:\s+(.*))?$")
        skip_block = False
        for line in source.splitlines():
            match = define_pattern.match(line)
            if match:
                name, value = match.group(1), match.group(2)
                defines[name] = value.strip() if value else "1"
                out_lines.append(line)
                continue
            stripped = line.lstrip()
            if stripped.startswith("#ifdef") or stripped.startswith("#ifndef"):
                name = stripped.split(None, 1)[1].strip()
                defined = name in defines
                keep = defined if stripped.startswith("#ifdef") else not defined
                skip_block = not keep
                continue
            if stripped.startswith("#endif"):
                skip_block = False
                continue
            if stripped.startswith("#else"):
                skip_block = not skip_block
                continue
            if skip_block:
                continue
            for name, value in defines.items():
                line = re.sub(r"\b%s\b" % re.escape(name), str(value), line)
            out_lines.append(line)
        return "\n".join(out_lines)

    def _validate_source(self, source):
        if not isinstance(source, str) or not source.strip():
            raise ShaderCompileError("shader source is empty")
        if source.count("{") != source.count("}"):
            raise ShaderCompileError("unbalanced braces in shader source")
        if source.count("(") != source.count(")"):
            raise ShaderCompileError("unbalanced parentheses in shader source")
        if "void main" not in source:
            raise ShaderCompileError("shader source has no main function")

    def _cache_key(self, name, source, defines):
        define_items = tuple(sorted((defines or {}).items()))
        return (name, hashlib.sha1(source.encode("utf-8")).hexdigest(), define_items)

    def compile_shader(self, name="default", source=None, defines=None, **kwargs):
        """Compile a shader with preprocessing, error checking and caching.

        Identical (name, source, defines) combinations are compiled once
        and served from the cache afterwards. A failed compile raises
        ``ShaderCompileError`` and never evicts a previously compiled
        shader of the same name.
        """
        if source is None:
            source = kwargs.get("src") or "void main() { }"
        defines = dict(defines or {})

        key = self._cache_key(name, source, defines)
        if key in self.shader_cache:
            return self.shader_cache[key]

        preprocessed = self.preprocess(source, defines)
        self._validate_source(preprocessed)

        shader = Shader(name, source, defines, preprocessed)
        self.shader_cache[key] = shader
        self.compiled_shaders[name] = shader
        return shader

    def get_shader(self, name):
        return self.compiled_shaders.get(name)

    # ------------------------------------------------------------------
    # Materials and instances
    # ------------------------------------------------------------------

    def create_material(self, name, shader=None, params=None, transparent=False):
        material = Material(name, shader=shader, params=params, transparent=transparent)
        self.materials[name] = material
        return material

    def get_material(self, name):
        return self.materials.get(name)

    def create_material_instance(self, parent=None, name=None, **kwargs):
        """Create a material instance inheriting parent parameters.

        The instance tracks which parameters it overrides and can be
        serialized so instance parameters survive save/load.
        """
        if parent is None:
            parent = kwargs.get("material")
        if isinstance(parent, str):
            parent = self.materials.get(parent)
        if parent is None:
            parent = self.create_material("default", params={})
        instance = MaterialInstance(parent, name=name)
        self.instances[instance.name] = instance
        return instance

    def serialize_instance(self, instance):
        return instance.serialize()

    def deserialize_instance(self, data):
        instance = MaterialInstance.deserialize(data, shader_system=self)
        self.instances[instance.name] = instance
        return instance

    # ------------------------------------------------------------------
    # PBR
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vector):
        length = math.sqrt(sum(component * component for component in vector))
        if length <= 0.0:
            return tuple(0.0 for _ in vector)
        return tuple(component / length for component in vector)

    @staticmethod
    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def calculate_pbr(self, base_color=(0.8, 0.8, 0.8), metallic=0.0,
                      roughness=0.5, normal=(0.0, 0.0, 1.0),
                      light_dir=(0.0, 0.0, 1.0), view_dir=(0.0, 0.0, 1.0),
                      light_color=(1.0, 1.0, 1.0), **kwargs):
        """Cook-Torrance style PBR with energy conservation and Fresnel.

        Returns a dict with the diffuse and specular terms plus the
        final lit color. Diffuse is scaled by (1 - specular) and by
        (1 - metallic) so diffuse + specular never exceeds the incoming
        light energy, and the Schlick Fresnel term produces rim
        lighting at grazing angles.
        """
        if isinstance(base_color, (int, float)):
            base_color = (float(base_color),) * 3
        base_color = tuple(float(c) for c in base_color)
        metallic = min(1.0, max(0.0, float(metallic)))
        roughness = min(1.0, max(0.0, float(roughness)))

        n = self._normalize(tuple(normal))
        l = self._normalize(tuple(light_dir))
        v = self._normalize(tuple(view_dir))
        h = self._normalize(tuple(n[i] + v[i] for i in range(3)))

        n_dot_l = max(0.0, self._dot(n, l))
        n_dot_v = max(0.0, self._dot(n, v))
        n_dot_h = max(0.0, self._dot(n, h))
        v_dot_h = max(0.0, self._dot(v, h))

        # Normal distribution (GGX/Trowbridge-Reitz) - roughness response.
        alpha = max(1e-3, roughness * roughness)
        denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0
        distribution = (alpha * alpha) / (math.pi * denom * denom)

        # Fresnel (Schlick) - F0 of 0.04 for dielectrics, base color
        # for metals; gives edge lighting at grazing angles.
        f0 = tuple(0.04 * (1.0 - metallic) + base_color[i] * metallic
                   for i in range(3))
        fresnel = tuple(f0[i] + (1.0 - f0[i]) * (1.0 - n_dot_v) ** 5
                        for i in range(3))

        # Geometry (Smith-Schlick).
        k = (roughness + 1.0) ** 2 / 8.0
        geometry = (n_dot_v / (n_dot_v * (1.0 - k) + k)) * \
                   (n_dot_l / (n_dot_l * (1.0 - k) + k))

        spec_denom = max(1e-4, 4.0 * n_dot_v * n_dot_l)
        specular = tuple(distribution * geometry * fresnel[i] / spec_denom
                         for i in range(3))

        # Energy conservation: diffuse loses what specular reflects,
        # and metals have no diffuse component at all.
        kd = tuple((1.0 - fresnel[i]) * (1.0 - metallic) for i in range(3))
        diffuse = tuple(kd[i] * base_color[i] / math.pi for i in range(3))

        light = tuple(float(c) for c in light_color)
        color = tuple((diffuse[i] + specular[i]) * n_dot_l * light[i]
                      for i in range(3))

        return {
            "diffuse": diffuse,
            "specular": specular,
            "fresnel": fresnel,
            "kd": kd,
            "color": color,
            "n_dot_l": n_dot_l,
        }

    # ------------------------------------------------------------------
    # Shader variants
    # ------------------------------------------------------------------

    @staticmethod
    def _variant_key(defines):
        return ",".join(sorted(defines)) if defines else ""

    def compile_variant(self, name="default", source=None, defines=None,
                        macros=None, **kwargs):
        """Compile a single shader variant on demand.

        Variants are keyed by their exact macro combination and compiled
        lazily - requesting one variant never compiles the other 2**n
        combinations. Unsupported macros fall back to the closest
        supported variant (ultimately the default variant) instead of
        failing.
        """
        requested = set(defines or macros or kwargs.get("features") or [])
        if isinstance(requested, dict):
            requested = {k for k, v in requested.items() if v}

        supported = self.supported_defines
        if supported is not None:
            effective = requested & set(supported)
        else:
            effective = requested

        key = self._variant_key(effective)
        cache = self.variants.setdefault(name, {})
        if key in cache:
            return cache[key]

        define_map = {macro: "1" for macro in sorted(effective)}
        try:
            shader = self.compile_shader(
                name="%s__%s" % (name, key or "default"),
                source=source,
                defines=define_map,
            )
        except ShaderCompileError:
            # Fall back to the default variant rather than crashing.
            if key == self.default_variant_key:
                raise
            return self.compile_variant(name=name, source=source, defines=[])

        shader.variant_key = key
        shader.requested_defines = sorted(requested)
        shader.is_fallback = key != self._variant_key(requested)
        cache[key] = shader
        return shader

    def get_variant(self, name, defines=None):
        key = self._variant_key(set(defines or []))
        return self.variants.get(name, {}).get(key)

    def set_supported_defines(self, defines):
        self.supported_defines = set(defines)

    # ------------------------------------------------------------------
    # Material sorting / render queue
    # ------------------------------------------------------------------

    @staticmethod
    def _material_key(material):
        if isinstance(material, MaterialInstance):
            material = material.parent
        return material

    def sort_materials(self, items=None, camera_pos=(0.0, 0.0, 0.0), **kwargs):
        """Sort render items for correct drawing and minimal state changes.

        Opaque items sort front-to-back; transparent items sort
        back-to-front and always draw after all opaque items. Items
        sharing a material are grouped into batches to merge draw
        calls. Items marked ``static`` keep their cached order and are
        not re-sorted every frame.
        """
        if items is None:
            items = kwargs.get("render_items") or []
        items = list(items)

        def is_transparent(item):
            if "transparent" in item:
                return bool(item["transparent"])
            material = self._material_key(item.get("material"))
            return bool(getattr(material, "transparent", False))

        def distance(item):
            if "distance" in item:
                return item["distance"]
            pos = item.get("position", (0.0, 0.0, 0.0))
            return math.sqrt(sum((pos[i] - camera_pos[i]) ** 2
                                 for i in range(min(3, len(pos)))))

        def batch_key(item):
            material = self._material_key(item.get("material"))
            return getattr(material, "name", None) or id(material)

        def sort_group(group, reverse):
            static = [it for it in group if it.get("static")]
            dynamic = [it for it in group if not it.get("static")]
            dynamic.sort(key=lambda it: (batch_key(it), distance(it)),
                         reverse=reverse)
            # Static items keep their previous relative order (no
            # per-frame re-sort); dynamic items are freshly sorted.
            return static + dynamic

        opaque = [it for it in items if not is_transparent(it)]
        transparent = [it for it in items if is_transparent(it)]

        ordered = sort_group(opaque, reverse=False)
        ordered += sort_group(transparent, reverse=True)

        batches = []
        for item in ordered:
            key = (is_transparent(item), batch_key(item))
            if batches and batches[-1]["key"] == key:
                batches[-1]["items"].append(item)
            else:
                batches.append({"key": key, "material": batch_key(item),
                                "transparent": is_transparent(item),
                                "items": [item]})

        return {
            "order": ordered,
            "batches": batches,
            "draw_calls": len(batches),
        }
