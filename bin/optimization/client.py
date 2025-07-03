from framework.prelude import Settings
from framework.optimization import connect_to_server


def rosenbrock_function(individual):
    """
    Rosenbrock function: classic optimization benchmark.
    
    Global minimum: f(1, 1, ..., 1) = 0
    
    Args:
        individual: numpy array representing the solution vector
        
    Returns:
        float: objective function value
    """
    total = 0.0
    for i in range(individual.shape[0] - 1):
        total += 100.0 * (individual[i + 1] - individual[i] ** 2) ** 2 + (1 - individual[i]) ** 2
    return total


def main():
    settings = Settings()

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    connect_to_server(
        settings.Server.HOST,
        settings.Server.PORT,
        evaluation_function=rosenbrock_function,
    )


if __name__ == "__main__":
    main()