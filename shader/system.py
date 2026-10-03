"""
pairwise-shader-system - ShaderSystem

A fixed shader system implementation:

- compile_shader(): preprocessing (macros/#ifdef), error checking, caching
- create_material_instance(): param inheritance, override tracking, serialization
- calculate_pbr(): normal/roughness aware Cook-Torrance, energy conservation, Fresnel
- compile_variant(): on-demand variant compilation with platform fallback
- sort_materials(): transparency/distance sorting, batching, static caching
"""

from __future__ import annotations

import hashlib
import math
import re


class ShaderCompileError(Exception):
    """Raised when shader compilation fails."""


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _normalize(vec):
    length = math.sqrt(sum(c * c for c in vec))
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(c / length for c in vec)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class CompiledShader:
    """A successfully preprocessed, validated and cached shader."""

    def __init__(self, source, macros, preprocessed, key):
        self.source = source
        self.macros = dict(macros)
        self.preprocessed_source = preprocessed
        self.key = key
        self.compiled = True

    def __repr__(self):
        return f"CompiledShader(key={self.key!r}, macros={self.macros!r})"


class Material:
    """Base material with a live parameter dictionary."""

    def __init__(self, name="material", **params):
        self.name = name
        self.params = dict(params)

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def set_param(self, name, value):
        self.params[name] = value

    def serialize(self):
        return {"type": "material", "name": self.name, "params": dict(self.params)}


class MaterialInstance:
    """Material instance: inherits parent params live, tracks its own overrides."""

    def __init__(self, parent, overrides=None, name=None):
        self.parent = parent
        self.name = name or f"{getattr(parent, 'name', 'material')}_instance"
        self._overrides = dict(overrides or {})

    def get_param(self, name, default=None):
        if name in self._overrides:
            return self._overrides[name]
        if hasattr(self.parent, "get_param"):
            return self.parent.get_param(name, default)
        return default

    def set_param(self, name, value):
        self._overrides[name] = value

    def clear_override(self, name):
        self._overrides.pop(name, None)

    def is_overridden(self, name):
        return name in self._overrides

    @property
    def overridden_params(self):
        return dict(self._overrides)

    @property
    def params(self):
        merged = dict(getattr(self.parent, "params", {}))
        merged.update(self._overrides)
        return merged

    def serialize(self):
        parent_name = getattr(self.parent, "name", None)
        return {
            "type": "material_instance",
            "name": self.name,
            "parent": parent_name,
            "overrides": dict(self._overrides),
        }


_DEFAULT_SHADER_SOURCE = """\
#ifdef USE_LIGHTING
#define LIGHT_COUNT 4
#endif
void main() {
    gl_Position = vec4(0.0);
}
"""


class ShaderSystem:
    def __init__(self):
        self.state = {}
        self._shader_cache = {}
        self._variant_cache = {}
        self._materials = {}
        self._supported_macros = None  # None -> everything supported
        self._sort_cache = None
        self._sort_dirty = True

    # ------------------------------------------------------------------
    # Shader compilation: preprocessing + error checking + caching
    # ------------------------------------------------------------------
    def _preprocess(self, source, macros):
        defines = {str(k): str(v) for k, v in macros.items()}
        output = []
        # stack entries: [parent_active, branch_taken]
        cond_stack = []
        active = True

        for lineno, raw_line in enumerate(source.splitlines(), start=1):
            stripped = raw_line.strip()
            directive = stripped.split(None, 1)[0] if stripped.startswith("#") else ""

            if directive == "#ifdef":
                name = stripped.split(None, 1)[1].strip()
                cond = name in defines
                cond_stack.append([active, cond])
                active = active and cond
                continue
            if directive == "#ifndef":
                name = stripped.split(None, 1)[1].strip()
                cond = name not in defines
                cond_stack.append([active, cond])
                active = active and cond
                continue
            if directive == "#else":
                if not cond_stack:
                    raise ShaderCompileError(f"#else without #ifdef at line {lineno}")
                parent_active, taken = cond_stack[-1]
                active = parent_active and not taken
                cond_stack[-1][1] = True
                continue
            if directive == "#endif":
                if not cond_stack:
                    raise ShaderCompileError(f"#endif without #ifdef at line {lineno}")
                parent_active, _ = cond_stack.pop()
                active = parent_active
                continue

            if not active:
                continue

            if directive == "#define":
                parts = stripped.split(None, 2)
                if len(parts) < 2:
                    raise ShaderCompileError(f"malformed #define at line {lineno}")
                defines[parts[1]] = parts[2] if len(parts) > 2 else "1"
                continue

            line = raw_line
            for macro_name, macro_value in defines.items():
                line = re.sub(rf"\b{re.escape(macro_name)}\b", macro_value, line)
            output.append(line)

        if cond_stack:
            raise ShaderCompileError("unterminated #ifdef block")
        return "\n".join(output)

    def _validate(self, preprocessed):
        if not preprocessed.strip():
            raise ShaderCompileError("shader source is empty after preprocessing")
        for open_ch, close_ch in (("{", "}"), ("(", ")")):
            if preprocessed.count(open_ch) != preprocessed.count(close_ch):
                raise ShaderCompileError(
                    f"unbalanced {open_ch}{close_ch} in shader source"
                )
        if "main" not in preprocessed:
            raise ShaderCompileError("shader has no main entry point")

    def compile_shader(self, source=None, macros=None, **kwargs):
        """Preprocess, validate and compile a shader, with caching.

        Repeated calls with the same source+macros return the cached
        CompiledShader instead of recompiling. Failures raise
        ShaderCompileError and never clobber the cached good shader.
        """
        if source is None:
            source = _DEFAULT_SHADER_SOURCE
        macros = dict(macros or {})
        macros.update({str(k): v for k, v in kwargs.items()})

        key_material = (source + "\0" + repr(sorted(macros.items()))).encode("utf-8")
        key = hashlib.sha1(key_material).hexdigest()

        cached = self._shader_cache.get(key)
        if cached is not None:
            return cached

        preprocessed = self._preprocess(source, macros)
        self._validate(preprocessed)

        shader = CompiledShader(source, macros, preprocessed, key)
        self._shader_cache[key] = shader
        self.state["last_compiled"] = key
        return shader

    # ------------------------------------------------------------------
    # Material instances: inheritance + override tracking + serialization
    # ------------------------------------------------------------------
    def create_material(self, name=None, **params):
        name = name or f"material_{len(self._materials)}"
        material = Material(name, **params)
        self._materials[name] = material
        return material

    def create_material_instance(self, parent=None, overrides=None, name=None):
        """Create an instance that inherits parent params live.

        Instance overrides are tracked separately, and serialize()
        persists only the overrides so they survive save/load.
        """
        if parent is None:
            parent = self.create_material()
        elif isinstance(parent, str):
            parent = self._materials[parent]
        elif isinstance(parent, dict):
            parent = self.create_material(**parent)
        instance = MaterialInstance(parent, overrides=overrides, name=name)
        self._materials[instance.name] = instance
        return instance

    def deserialize_material(self, data):
        """Rebuild a material or instance from serialized data."""
        data = dict(data)
        kind = data.get("type", "material")
        if kind == "material_instance":
            parent_name = data.get("parent")
            parent = self._materials.get(parent_name)
            if parent is None:
                parent = self.create_material(name=parent_name or "material")
            return self.create_material_instance(
                parent=parent, overrides=data.get("overrides"), name=data.get("name")
            )
        return self.create_material(name=data.get("name"), **data.get("params", {}))

    # ------------------------------------------------------------------
    # PBR: normal + roughness aware, energy conserving, with Fresnel
    # ------------------------------------------------------------------
    def calculate_pbr(
        self,
        normal=(0.0, 0.0, 1.0),
        light_dir=(0.0, 0.0, 1.0),
        view_dir=(0.0, 0.0, 1.0),
        albedo=(1.0, 1.0, 1.0),
        roughness=0.5,
        metallic=0.0,
    ):
        """Cook-Torrance GGX BRDF with Schlick Fresnel.

        Returns diffuse/specular/fresnel terms plus the final color.
        Energy conservation: kd = (1 - F) * (1 - metallic), so
        diffuse + specular never exceeds the incoming energy.
        """
        n = _normalize(tuple(normal))
        l = _normalize(tuple(light_dir))
        v = _normalize(tuple(view_dir))
        # half vector between light and view
        h = _normalize(tuple(l[i] + v[i] for i in range(3)))

        if isinstance(albedo, (int, float)):
            albedo = (float(albedo),) * 3
        albedo = tuple(float(c) for c in albedo)
        roughness = _clamp(float(roughness), 0.045, 1.0)
        metallic = _clamp(float(metallic))

        n_dot_l = max(_dot(n, l), 0.0)
        n_dot_v = max(_dot(n, v), 0.0)
        n_dot_h = max(_dot(n, h), 0.0)
        v_dot_h = max(_dot(v, h), 0.0)

        # F0: dielectrics ~0.04, metals use albedo
        f0 = tuple(0.04 * (1.0 - metallic) + c * metallic for c in albedo)

        # Schlick Fresnel -> grazing angles get rim lighting
        fresnel = tuple(
            f0c + (1.0 - f0c) * (1.0 - v_dot_h) ** 5 for f0c in f0
        )

        # GGX/Trowbridge-Reitz normal distribution (roughness matters)
        a = roughness * roughness
        a2 = a * a
        denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
        distribution = a2 / (math.pi * denom * denom + 1e-12)

        # Smith-Schlick geometry term
        k = (roughness + 1.0) ** 2 / 8.0

        def geometry_schlick(cos_theta):
            return cos_theta / (cos_theta * (1.0 - k) + k + 1e-12)

        geometry = geometry_schlick(n_dot_v) * geometry_schlick(n_dot_l)

        specular = tuple(
            (distribution * geometry * f)
            / (4.0 * n_dot_v * n_dot_l + 1e-6)
            for f in fresnel
        )

        # Energy conservation: diffuse only gets what specular didn't use
        kd = tuple((1.0 - f) * (1.0 - metallic) for f in fresnel)
        diffuse = tuple(kdc * c / math.pi for kdc, c in zip(kd, albedo))

        color = tuple(
            (d + s) * n_dot_l for d, s in zip(diffuse, specular)
        )

        return {
            "diffuse": diffuse,
            "specular": specular,
            "fresnel": fresnel,
            "color": color,
            "n_dot_l": n_dot_l,
            "roughness": roughness,
            "metallic": metallic,
        }

    # ------------------------------------------------------------------
    # Shader variants: on-demand compilation + platform fallback
    # ------------------------------------------------------------------
    def set_supported_macros(self, macros):
        """Declare which feature macros the current platform supports."""
        self._supported_macros = {str(m) for m in macros} if macros is not None else None

    def _unsupported(self, macros):
        if self._supported_macros is None:
            return set()
        return {m for m in macros if m not in self._supported_macros}

    def compile_variant(self, source=None, macros=None, _fell_back=False):
        """Compile exactly the requested variant (on demand).

        Variants are compiled only when requested and cached by macro
        combination, so N macros never explode into 2**N compiles.
        If the platform doesn't support a requested macro, the
        unsupported macros are stripped and a fallback variant is
        compiled instead of crashing.
        """
        macros = dict(macros or {})
        key = frozenset((str(k), str(v)) for k, v in macros.items())

        cached = self._variant_cache.get(key)
        if cached is not None:
            return cached

        unsupported = self._unsupported(macros)
        if unsupported:
            fallback_macros = {
                k: v for k, v in macros.items() if k not in unsupported
            }
            shader = self.compile_variant(
                source=source, macros=fallback_macros, _fell_back=True
            )
            self._variant_cache[key] = shader
            return shader

        shader = self.compile_shader(source=source, macros=macros)
        variant = {
            "shader": shader,
            "macros": macros,
            "fallback": _fell_back,
        }
        self._variant_cache[key] = variant
        return variant

    @property
    def compiled_variant_count(self):
        return len(self._variant_cache)

    # ------------------------------------------------------------------
    # Material sorting: transparency + distance, batching, static cache
    # ------------------------------------------------------------------
    def mark_sort_dirty(self):
        """Invalidate the static sort cache (e.g. objects moved)."""
        self._sort_dirty = True

    @staticmethod
    def _is_transparent(item):
        if item.get("transparent"):
            return True
        return float(item.get("opacity", 1.0)) < 1.0

    def sort_materials(self, renderables=None):
        """Sort renderables for drawing.

        - Opaque: front-to-back, grouped by material for batching.
        - Transparent: back-to-front by distance.
        - Identical consecutive materials merge into one draw call.
        - Fully static scenes are cached and not re-sorted every frame
          unless mark_sort_dirty() is called.
        """
        if renderables is None:
            renderables = []
        renderables = [dict(r) for r in renderables]

        all_static = all(r.get("static", False) for r in renderables)
        if all_static and not self._sort_dirty and self._sort_cache is not None:
            return self._sort_cache

        opaque = [r for r in renderables if not self._is_transparent(r)]
        transparent = [r for r in renderables if self._is_transparent(r)]

        # Opaque: batch-friendly (material first), then front-to-back.
        opaque.sort(key=lambda r: (r.get("material", ""), float(r.get("distance", 0.0))))
        # Transparent: strictly back-to-front.
        transparent.sort(key=lambda r: -float(r.get("distance", 0.0)))

        ordered = opaque + transparent

        batches = []
        for item in ordered:
            material = item.get("material", "")
            if (
                batches
                and batches[-1]["material"] == material
                and not self._is_transparent(item)
            ):
                batches[-1]["items"].append(item)
            else:
                batches.append({"material": material, "items": [item]})

        result = {
            "order": ordered,
            "batches": batches,
            "draw_calls": len(batches),
        }

        if all_static:
            self._sort_cache = result
            self._sort_dirty = False
        else:
            self._sort_cache = None
        return result
