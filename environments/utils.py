import numpy
from studyLib import optimizer, miscellaneous, wrap_mjc


def show_mujoco_env(
        env: optimizer.MuJoCoEnvInterface,
        para,
        window: miscellaneous.Window,
        camera: wrap_mjc.Camera,
        width: int = 640,
        height: int = 480
) -> float:
    window.set_recorder(miscellaneous.Recorder("result.mp4", 30, width, height))
    return env.calc_and_show(para, window, camera)


def cmaes_optimize(
        generation: int,
        population: int,
        mu: int,
        sigma: float,
        env_creator: optimizer.MuJoCoEnvCreator,
        max_thread: int = 4,
        minimalize: bool = True,
        window_and_camera: (miscellaneous.Window, wrap_mjc.Camera) = None
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.CMAES(env_creator.dim(), generation, population, mu, sigma, minimalize, max_thread)
    if window_and_camera is None:
        opt.optimize(env_creator)
    else:
        window, camera = window_and_camera
        opt.optimize_with_recoding_min(env_creator, window, camera)
    return opt.get_best_para(), opt.get_history()


def cmaes_optimize_server(
        generation: int,
        population: int,
        mu: int,
        sigma: float,
        env_creator: optimizer.MuJoCoEnvCreator,
        port: int = 52325,
        minimalize: bool = True,
        window_and_camera: (miscellaneous.Window, wrap_mjc.Camera) = None
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.ServerCMAES(port, env_creator.dim(), generation, population, mu, sigma, minimalize)
    if window_and_camera is None:
        opt.optimize(env_creator)
    else:
        window, camera = window_and_camera
        opt.optimize_with_recoding_min(env_creator, window, camera)
    return opt.get_best_para(), opt.get_history()


def _client_proc(proc_id: int, default_env_creator: optimizer.EnvCreator, address: str, port: int, buf_size: int):
    opt = optimizer.ClientCMAES(address, port, buf_size)
    print(f"Start THREAD{proc_id}")

    # window = miscellaneous.Window(640, 480)
    # camera = wrap_mjc.Camera((0, 350, 0), 1200, 0, 90)

    error_count = 0
    while True:
        print(f"THREAD{proc_id} is calculating")
        result, pe = opt.optimize(default_env_creator)
        # result, pe = opt.optimize_and_show(default_env_creator, window, camera)  # debug

        if result is optimizer.ClientCMAES.Result.Succeed:
            error_count = 0
            continue
        elif result is optimizer.ClientCMAES.Result.ErrorOccurred:
            if error_count >= 3:
                print(f"THREAD{proc_id} failed to reconnect the server. : ", pe)
                break
            print(f"THREAD{proc_id} is retrying.")
            error_count += 1
            continue
        elif result is optimizer.ClientCMAES.Result.FatalErrorOccurred:
            print(f"THREAD{proc_id} face fatal error : ", pe)
            break


def cmaes_optimize_client(
        num_thread: int,
        default_env_creator: optimizer.EnvCreator,
        address: str,
        buf_size: int = 3072,
        port: int = 52325,
):
    from multiprocessing import Process

    process_list = []
    for proc_id in range(num_thread):
        process = Process(
            target=_client_proc,
            args=(proc_id, default_env_creator, address, port, buf_size)
        )
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()
