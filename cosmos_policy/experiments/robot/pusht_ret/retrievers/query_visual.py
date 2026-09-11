"""Render a separate canonical agent appearance for online retrieval only."""

import numpy as np


QUERY_VISUAL_MODES = ("none", "blue_circle")


def retrieval_query_frame(env, frame, mode="none"):
    """Keep policy pixels/physics intact; reuse demo drawing at radius 15.

    This is a simulator rendering intervention, not a pixel-only detector. It
    reuses the current scene without advancing it or changing collision shapes.
    """
    if mode not in QUERY_VISUAL_MODES:
        raise ValueError(f"Unknown retrieval query visual mode: {mode}")
    if mode == "none":
        return frame
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected uint8 RGB query frame")

    import pymunk
    from pymunk.space_debug_draw_options import SpaceDebugColor
    from gym_pusht.envs.pymunk_override import DrawOptions

    scene = env.unwrapped
    shapes = list(scene.agent.shapes)
    if len(shapes) != 1:
        raise ValueError("Expected a single PushT agent shape")
    shape = shapes[0]
    if isinstance(shape, pymunk.Circle):
        if shape.radius != 15 or tuple(shape.color)[:3] != (65, 105, 225):
            raise ValueError("Unsupported circle agent appearance")
        return frame.copy()
    if not isinstance(shape, pymunk.Poly) or len(shape.get_vertices()) != 3:
        raise ValueError("Expected the PushT triangle agent")
    vertices = np.asarray([scene.agent.local_to_world(v) for v in shape.get_vertices()])

    class QueryDrawOptions(DrawOptions):
        replacements = 0

        def draw_polygon(self, verts, radius, outline_color, fill_color):
            if len(verts) == 3 and np.allclose(verts, vertices, atol=1e-4, rtol=0):
                blue = SpaceDebugColor(65, 105, 225, 255)
                self.draw_circle(scene.agent.position, 0, 15, blue, blue)
                self.replacements += 1
            else:
                super().draw_polygon(verts, radius, outline_color, fill_color)

    options = []

    def factory(surface):
        result = QueryDrawOptions(surface)
        options.append(result)
        return result

    screen = scene._draw(draw_options_factory=factory)
    if len(options) != 1 or options[0].replacements != 1:
        raise RuntimeError("Query renderer did not replace exactly one agent")
    return scene._get_img(screen, width=frame.shape[1], height=frame.shape[0])
