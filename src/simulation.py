"""MuJoCo simulation wrapper: model loading, reset, stepping, viewer, screenshots."""
from __future__ import annotations

import shutil
import struct
import subprocess
import time
import zlib
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "trustloop_scene.xml"


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal dependency-free PNG writer for an HxWx3 uint8 image."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


class Simulation:
    def __init__(self, scene_path: str | Path = DEFAULT_SCENE, viewer: bool = False,
                 realtime: bool = True, screenshot_dir: str | Path | None = None):
        self.scene_path = Path(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime and viewer
        self.viewer = None
        self._viewer_requested = viewer
        self._last_sync = 0.0
        self._wall_start = None
        self._sim_start = 0.0
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self._renderer = None
        self._renderer_failed = False
        self._video = None          # (ffmpeg process, renderer, frame period, next frame time, path)
        self.home_key = self.model.key("home").id
        self.cube_body = self.model.body("cube").id
        self.cube_qadr = self.model.jnt_qposadr[self.model.joint("cube_free").id]
        self.cube_dadr = self.model.jnt_dofadr[self.model.joint("cube_free").id]

    # ------------------------------------------------------------------ setup
    def reset(self, cube_pos=None, cube_yaw: float = 0.0, settle_steps: int = 250) -> None:
        """Reset to the HOME keyframe; optionally place the cube somewhere else first."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        if cube_pos is not None:
            self.set_cube_pose(cube_pos, cube_yaw)
        mujoco.mj_forward(self.model, self.data)
        # Let contacts settle so every run starts from a physically resting state.
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        self._wall_start = None

    def set_cube_pose(self, pos, yaw: float = 0.0) -> None:
        """Initial-condition setter (used for resets and verifier tests only,
        never during a manipulation task)."""
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[self.cube_dadr:self.cube_dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def open_viewer(self) -> None:
        if not self._viewer_requested or self.viewer is not None:
            return
        import mujoco.viewer  # imported lazily so headless runs need no GL
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Free (orbitable) camera framed like the "overview" camera.
        self.viewer.cam.lookat[:] = [0.30, 0.02, 0.50]
        self.viewer.cam.distance = 1.75
        self.viewer.cam.azimuth = 155
        self.viewer.cam.elevation = -32

    def start_recording(self, path: str | Path, fps: int = 30, width: int = 960, height: int = 540) -> bool:
        """Record an MP4 of the overview camera (needs ffmpeg on PATH + offscreen GL)."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("[video] ffmpeg not found on PATH - recording disabled")
            return False
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
        except Exception as exc:
            print(f"[video] offscreen rendering unavailable: {exc!r}")
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", str(path)],
            stdin=subprocess.PIPE)
        self._video = [proc, renderer, 1.0 / fps, self.data.time, path]
        return True

    def stop_recording(self) -> Path | None:
        if self._video is None:
            return None
        proc, renderer, _, _, path = self._video
        self._video = None
        proc.stdin.close()
        proc.wait()
        renderer.close()
        return path

    def _record_frame(self) -> None:
        proc, renderer, period, next_t, _ = self._video
        if self.data.time + 1e-9 < next_t:
            return
        renderer.update_scene(self.data, camera="overview")
        proc.stdin.write(renderer.render().tobytes())
        self._video[3] = next_t + period

    def close(self) -> None:
        self.stop_recording()
        if self.viewer is not None:
            self.viewer.close()
            # The viewer tears down its GL context on its own thread; exiting the
            # interpreter before that finishes can crash at shutdown. Wait for it.
            t0 = time.perf_counter()
            while self.viewer.is_running() and time.perf_counter() - t0 < 3.0:
                time.sleep(0.05)
            time.sleep(0.5)
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --------------------------------------------------------------- stepping
    @property
    def time(self) -> float:
        return float(self.data.time)

    def viewer_alive(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self._video is not None:
            self._record_frame()
        if self.viewer is not None:
            now = time.perf_counter()
            if self.realtime:
                if self._wall_start is None:
                    self._wall_start, self._sim_start = now, self.data.time
                ahead = (self.data.time - self._sim_start) - (now - self._wall_start)
                if ahead > 0:
                    time.sleep(ahead)
            if now - self._last_sync > 1 / 60:
                self.viewer.sync()
                self._last_sync = now

    # ------------------------------------------------------------- state I/O
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def cube_rot(self) -> np.ndarray:
        return self.data.xmat[self.cube_body].reshape(3, 3).copy()

    def cube_yaw(self) -> float:
        r = self.cube_rot()
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def cube_vel(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.data.qvel[self.cube_dadr:self.cube_dadr + 6]
        return v[:3].copy(), v[3:].copy()

    def site_pos(self, name: str = "ee_site") -> np.ndarray:
        return self.data.site(name).xpos.copy()

    # ------------------------------------------------------------ screenshots
    def screenshot(self, name: str, camera: str = "overview") -> Path | None:
        if self.screenshot_dir is None or self._renderer_failed:
            return None
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
            self._renderer.update_scene(self.data, camera=camera)
            img = self._renderer.render()
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.screenshot_dir / f"{name}.png"
            write_png(path, img)
            return path
        except Exception as exc:  # no GL context available, etc.
            self._renderer_failed = True
            print(f"[screenshot] disabled: {exc!r} (set MUJOCO_GL=egl or osmesa for headless rendering)")
            return None
