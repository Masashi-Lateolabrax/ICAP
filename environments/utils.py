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


def _client_proc(proc_id: int, init_env: optimizer.EnvInterface, address: str, port: int, buf_size: int):
    import copy

    env = copy.deepcopy(init_env)
    opt = optimizer.ClientCMAES(address, port, buf_size)
    print(f"Start THREAD{proc_id}")

    error = None
    can_recover = True
    err_count = 0
    while (error is None or can_recover) and err_count < 5:
        print(f"THREAD{proc_id} is calculating")
        error, can_recover = opt.optimize(env)
        if error is not None and can_recover:
            err_count += 1
        else:
            err_count = 0

    print(f"THREAD{proc_id} is close : ", error)


def cmaes_optimize_client(
        num_thread: int,
        init_env: optimizer.EnvInterface,
        address: str,
        buf_size: int = 3072,
        port: int = 52325,
):
    from multiprocessing import Process

    process_list = []
    for proc_id in range(num_thread):
        process = Process(
            target=_client_proc,
            args=(proc_id, init_env, address, port, buf_size)
        )
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()
