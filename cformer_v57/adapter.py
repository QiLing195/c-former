from __future__ import annotations

from cformer_v55 import CognitiveMemory


class CognitiveTextAdapter:
    """Renders V5.5 canonical memory into auditable retrieval documents."""

    def render(self, memory: CognitiveMemory, version: int | None = None) -> tuple[str, ...]:
        objects = {obj.object_id: obj for obj in memory.snapshot_objects(version)}
        documents: list[str] = []
        for state in memory.snapshot_states(version):
            obj = objects[state.object_id]
            aliases = ", ".join(sorted(obj.aliases))
            properties = "; ".join(f"{field}={value}" for field, value in state.properties)
            documents.append(
                f"对象 {obj.canonical_name}; 别名 {aliases}; 类型 {obj.object_type}; "
                f"状态 {properties}; 有效期 {state.valid_from} 到 {state.valid_to}; "
                f"证据 {state.state_id}。"
            )
        for transformation in memory.snapshot_transformations(version):
            source = ", ".join(objects[item].canonical_name for item in transformation.input_object_ids)
            target = objects[transformation.output_object_id].canonical_name
            constraints = "; ".join(
                f"{item.field} {item.operator.value} {item.expected}"
                for item in transformation.constraints
            )
            documents.append(
                f"Transformation {transformation.operator}; 输入 {source}; 输出 {target}; "
                f"条件 {constraints}; 证据 {transformation.transformation_id}。"
            )
        return tuple(documents)
