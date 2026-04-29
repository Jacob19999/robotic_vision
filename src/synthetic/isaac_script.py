from __future__ import annotations

from pathlib import Path

from src.synthetic.models import SyntheticGenerationPlan


def render_isaac_replicator_script(plan: SyntheticGenerationPlan) -> str:
    """Render a small Isaac Sim script for the Phase 2 plan.

    The generated script is intended to run inside Isaac Sim's Python environment.
    It deliberately keeps one RGB camera and tight 2D boxes so it fits the 12 GB
    workstation constraint described in the project README.
    """

    plan_json = plan.model_dump_json(indent=2)
    return f'''from __future__ import annotations

import json
import random
from pathlib import Path

import omni.replicator.core as rep


PLAN = json.loads(r"""{plan_json}""")


def _pick_range(values):
    return random.uniform(float(values[0]), float(values[1]))


def _make_light(scene):
    intensity = _pick_range(scene["lighting_intensity_range"])
    return rep.create.light(light_type="Distant", intensity=intensity, rotation=(315, 0, random.uniform(0, 360)))


def _make_camera(scene):
    distance = _pick_range(scene["camera_distance_range_m"])
    height = _pick_range(scene["camera_height_range_m"])
    return rep.create.camera(
        position=(random.uniform(-0.3, 0.3), -distance, height),
        look_at=(0, 0, 0.35),
        focal_length=random.choice([24, 28, 35]),
    )


def _instantiate_usd(path, semantic_label=None):
    node = rep.create.from_usd(path)
    if semantic_label:
        node.set_semantics([("class", semantic_label)])
    return node


def _scatter_objects(scene):
    objects = []
    for class_id in scene["target_classes"]:
        candidates = scene["object_usd_paths"].get(class_id, [])
        if not candidates:
            continue
        node = _instantiate_usd(random.choice(candidates), class_id)
        with node:
            rep.modify.pose(
                position=(random.uniform(-0.45, 0.45), random.uniform(-0.25, 0.25), random.uniform(0.05, 0.3)),
                rotation=(0, 0, random.uniform(0, 360)),
                scale=random.uniform(0.85, 1.15),
            )
        objects.append(node)

    clutter_min, clutter_max = scene["clutter_count_range"]
    for _ in range(random.randint(int(clutter_min), int(clutter_max))):
        if not scene["distractor_usd_paths"]:
            break
        node = _instantiate_usd(random.choice(scene["distractor_usd_paths"]))
        with node:
            rep.modify.pose(
                position=(random.uniform(-0.6, 0.6), random.uniform(-0.35, 0.35), random.uniform(0.03, 0.25)),
                rotation=(0, 0, random.uniform(0, 360)),
                scale=random.uniform(0.6, 1.4),
            )
        objects.append(node)
    return objects


def run():
    random.seed(int(PLAN["seed"]))
    output_root = Path(PLAN["output_root"]).resolve()

    for scene in PLAN["scenes"]:
        scene_output = output_root / scene["scene_id"]
        camera = _make_camera(scene)
        render_product = rep.create.render_product(
            camera,
            (int(scene["resolution_width"]), int(scene["resolution_height"])),
        )
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(scene_output),
            rgb=True,
            bounding_box_2d_tight=True,
        )
        writer.attach([render_product])

        with rep.trigger.on_frame(num_frames=int(scene["frame_count"])):
            _make_light(scene)
            _scatter_objects(scene)

        rep.orchestrator.run()


if __name__ == "__main__":
    run()
'''


def write_isaac_replicator_script(plan: SyntheticGenerationPlan, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_isaac_replicator_script(plan), encoding="utf-8")
    return path

