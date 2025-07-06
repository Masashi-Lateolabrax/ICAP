from framework.optimization import run_adaptive_client_manager
from settings import MySettings


def main():
    settings = MySettings()
    
    from src.main import evaluation_function
    
    run_adaptive_client_manager(
        host=settings.Server.HOST,
        port=settings.Server.PORT,
        evaluation_function=evaluation_function,
        min_processes=2
    )


if __name__ == "__main__":
    main()
