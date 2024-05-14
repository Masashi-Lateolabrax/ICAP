import os
import re
import random
import math
import subprocess

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip


def is_overlap(new_circle, circles, r):
    for circle in circles:
        dist = math.sqrt((new_circle[0] - circle[0]) ** 2 + (new_circle[1] - circle[1]) ** 2)
        if dist < 2 * r:
            return True
    return False


def generate_circles(n, r, area):
    circles = []
    attempts = 0
    max_attempts = 10

    while len(circles) < n and attempts < max_attempts:
        angle = random.uniform(0, 2 * math.pi)
        direction = random.uniform(0, 360)
        distance = random.uniform(0, area - r)
        new_circle = (distance * math.cos(angle), distance * math.sin(angle), direction)

        if not is_overlap(new_circle, circles, r):
            circles.append(new_circle)
            attempts = 0
            continue
        attempts += 1

    if attempts == max_attempts:
        print(f"Warning: Maximum attempts reached with {len(circles)} circles placed.")

    return circles


def get_latest_history(directory_path):
    def get_commit_timestamp(commit_hash):
        command = ['git', 'log', '-n', '1', '--pretty=format:%at', commit_hash]
        result = subprocess.run(command, cwd=directory_path, capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            return None

    file_names = os.listdir(directory_path)
    commit_timestamps = {}
    commit_hash_regex = re.compile(r'history_([a-f0-9]+)\.npz')

    for file_name in file_names:
        res = commit_hash_regex.match(file_name)
        if res:
            hash_part = res.group(1)
            timestamp = get_commit_timestamp(hash_part)
            if timestamp is not None:
                commit_timestamps[hash_part] = timestamp

    latest_commit_hash = max(commit_timestamps, key=commit_timestamps.get)
    latest_file = f'history_{latest_commit_hash}.npz'

    return os.path.join(directory_path, latest_file)


def get_current_history(directory_path):
    command = ['git', 'rev-parse', "HEAD"]
    cmd_result = subprocess.run(command, cwd=directory_path, capture_output=True, text=True)
    if cmd_result.returncode != 0:
        raise "Failed to exec the command."
    current_hash = cmd_result.stdout.strip()[0:8]

    result = os.path.join(directory_path, f'history_{current_hash}.npz')
    if not os.path.exists(result):
        print(os.path.abspath(result))
        raise f"Failed to find the history_XXXXXXXX.npz. Make sure to check out the correct commit."
    return result


def get_current_history(directory_path):
    command = ['git', 'rev-parse', "HEAD"]
    cmd_result = subprocess.run(command, cwd=directory_path, capture_output=True, text=True)
    if cmd_result.returncode != 0:
        raise "Failed to exec the command."
    current_hash = cmd_result.stdout.strip()[0:8]

    result = os.path.join(directory_path, f'history_{current_hash}.npz')
    if not os.path.exists(result):
        print(os.path.abspath(result))
        raise "Failed to find the history_XXXXXXXX.npz. Make sure to check out the correct commit."
    return result


def save_video(frames, videopath: str, framerate=60, allow_fps_reducing=True):
    """動画を保存する
- frames: ndarrayのlistか、ndarray
- framerate : 動画のfps
- videopath: 保存先のファイルパス、gif/mp4形式のみ
- allow_fps_reducing: gifとして保存する際にfpsを減少（frames枚数を減らす）させることを許すか"""
    # videopathがmp4のものかを確認
    ext = os.path.splitext(videopath)[1]
    assert ext in [".mp4", ".gif"]

    # 動画のサイズを取得
    if type(frames) == np.ndarray:
        _, height, width, _ = frames.shape
    else:
        height, width, _ = frames[0].shape

    if ext == ".mp4":
        _save_video_as_mp4(frames, videopath, framerate, width, height)
    else:
        save_video_as_gif(frames, videopath, framerate, allow_fps_reducing=allow_fps_reducing)


def _save_video_as_mp4(frames, videopath: str, framerate: int, width: int, height: int):
    # 動画をファイルとして保存
    video_writer = cv2.VideoWriter(videopath,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   framerate,
                                   (width, height))
    for frame in frames:
        # 色の並びをOpenCV用に変換
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(image)
    video_writer.release()


def save_video_as_gif(frames, videopath: str, fps: int, gif_fps: int = 6, *,
                      slow_down_fps: float = 1,
                      decreased_colors: int = -1,
                      allow_overwrite=False, allow_fps_reducing: bool = False):
    """動画をgifとして保存する
- frames: ndarrayのlistか、ndarray
- videopath: 保存先のファイルパス、gif形式のみ
- fps     : 元動画のfps
- gif_fps : 保存するgifのfps
- slow_down_fps     : 1より大きい場合、動画の速度を1/nにする (2の場合、gifの動画fpsを1/2にする)
- decreased_colors  : 1以上の場合、指定された数に減色する (16未満くらいが望ましい; K-Meansのクラスタ数を指定するため、数が大きいほど時間がかかる)
- allow_overwrite   : Falseの場合、名前の後ろに数字 ("_n") を付加する
- allow_fps_reducing: gifとして保存する際にfpsを減少 (frames枚数を減らす) させることを許すか

推奨設定は以下の通り

- gif_fps = 6
- slow_down_fps = 2 (1などでもよい。1未満は非推奨)
- decreased_colors = 12
- allow_overwrite = True
- allow_fps_reducing = True"""
    assert os.path.splitext(videopath)[1] == ".gif"
    print("gifに変換中...")
    fps /= slow_down_fps

    # フレーム数を減らす
    if allow_fps_reducing and (fps > gif_fps):
        step = int(fps / gif_fps)
        frames = frames[::step]
    else:
        gif_fps = min(int(fps), gif_fps)

    # 減色する
    if decreased_colors >= 1:
        frames = decrease_colors_by_k_means(frames, decreased_colors)

    # ファイル名を決定する
    newname = os.path.splitext(videopath)[0] + ".gif"
    if not allow_overwrite:
        newname = append_index_num_to_avoid_duplication(newname)
    # 保存する
    clip = ImageSequenceClip(frames, fps=gif_fps)
    clip.write_gif(newname)


def decrease_colors_by_k_means(frames: list[np.ndarray], clusters: int = 8):
    """K-Means法により画像群の減色を行う
- clusters : クラスター数 (処理時間を考えると16未満くらいが望ましい)"""
    # ndarray(y,x,[R,G,B]) を ndarray(y*x, [R,G,B]) に変形
    totalframe = np.concatenate([x.reshape((-1, 3)) for x in frames])
    totalframe = np.float32(totalframe)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(totalframe, clusters, None,
                                  criteria=criteria,
                                  attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    n_shift = 0
    for i, frame in enumerate(frames):
        n_pixels = frame.shape[0] * frame.shape[1]
        res = center[label[n_shift:n_shift + n_pixels].flatten()]
        frames[i] = res.reshape((frame.shape))
        n_shift += n_pixels

    return frames


def append_index_num_to_avoid_duplication(filename: str) -> str:
    """ファイル名の重複を避けるために、ファイル名の末尾に数字を付加する"""
    basename, ext = os.path.splitext(filename)
    newname = filename

    i = 1
    while True:
        if not os.path.exists(newname):
            return newname
        # 数字を変更
        newname = basename + f"_{i}" + ext
        i += 1
