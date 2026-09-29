import asyncio
import json
import logging
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import cloudpickle
import ray
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

from bioengine import __version__
from bioengine.cluster.ray_cluster import RayCluster
from bioengine.utils import check_permissions, create_logger


@ray.remote
def ray_task(func, args, kwargs):
    """
    Execute a Python function in an isolated Ray worker with comprehensive output capture.

    This is the core execution engine that runs user code in a distributed Ray
    environment. It acts like a "remote Python interpreter" that captures everything
    that happens during function execution, including output, errors, and results.

    Execution Environment:
    • Runs in completely separate Ray worker process
    • Captures all stdout/stderr output for streaming back to user
    • Handles both synchronous and asynchronous functions automatically
    • Provides consistent error reporting with full Python tracebacks

    Output Capture:
    Everything the function prints or writes to stderr is captured and returned,
    allowing users to see exactly what happened during execution, including
    progress updates, debug messages, and error output.

    Error Handling:
    Any exception that occurs during function execution is caught, formatted
    with a full Python traceback, and returned as part of the result dictionary
    rather than crashing the task.

    Args:
        func: The Python function object to execute (already loaded/deserialized)
        args: List of positional arguments to pass to the function
        kwargs: Dictionary of keyword arguments to pass to the function

    Returns:
        Dictionary containing execution results:
        • Success: {"result": return_value, "stdout": "...", "stderr": "..."}
        • Failure: {"error": "message", "traceback": "...", "stdout": "...", "stderr": "..."}

    Example:
        This function is called internally by Ray when executing code:
        ```python
        # Ray automatically handles the remote execution
        result = ray_task.remote(my_function, [arg1, arg2], {"param": "value"})
        output = await result  # Get the execution results
        ```

    Note:
        This is an internal Ray task function used by CodeExecutor.run_code().
        Users don't call this directly - it's invoked automatically by the Ray runtime.
    """
    import asyncio
    import contextlib
    import io
    import traceback

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            return {
                "result": result,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }


class CodeExecutor:
    """
    Secure Python code execution service for distributed computing environments.

    The CodeExecutor acts as a "code sandbox" that allows trusted administrators to run
    arbitrary Python functions in isolated Ray tasks. Think of it as a remote Python
    interpreter that runs your code safely in a distributed cluster while providing
    real-time monitoring and resource control.

    Why Use This Service:
    • Remote Execution: Run code on powerful cluster nodes instead of your local machine
    • Resource Control: Allocate specific CPU/GPU/memory resources for your computations
    • Output Streaming: See results and errors in real-time as your code runs
    • Security: Admin-only access with isolated execution environments
    • Flexibility: Support both quick code snippets and complex pre-built functions

    Common Use Cases:
    • Data Analysis: Run statistical computations on large datasets
    • Model Training: Execute machine learning workflows with GPU allocation
    • Batch Processing: Process files or data in parallel across cluster nodes
    • Experimentation: Test algorithms without local resource constraints
    • Debugging: Execute diagnostic code to troubleshoot distributed applications

    How It Works:
    1. Submit Python code or pre-serialized functions
    2. Specify resource requirements (CPUs, memory, GPUs)
    3. Code runs in isolated Ray task with timeout protection
    4. Results stream back with stdout/stderr in real-time
    5. Comprehensive error handling and execution logging

    Security Model:
    This service requires admin privileges as it executes arbitrary Python code.
    All execution happens in isolated Ray tasks, but should only be used by
    trusted administrators in controlled environments.

    Setup Example:
        ```python
        # Initialize and configure the executor
        executor = CodeExecutor(ray_cluster=cluster, debug=True)
        await executor.initialize(admin_users=["admin@company.com"])

        # Execute some data analysis code
        result = await executor.run_code(
            code=\"\"\"
            def analyze_sales(data):
                total = sum(data)
                average = total / len(data)
                return {"total": total, "average": average}
            \"\"\",
            function_name="analyze_sales",
            args=[[100, 200, 150, 300]],
            remote_options={"num_cpus": 2}
        )
        print(f"Analysis result: {result['result']}")
        ```
    """

    def __init__(
        self,
        ray_cluster: RayCluster,
        # Logger
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ) -> None:
        """
        Set up a new CodeExecutor instance with logging and cluster configuration.

        This constructor prepares the executor but doesn't set up admin permissions yet.
        You'll need to call initialize() after construction to specify which users
        can execute code through this service.

        Logging Configuration:
        • debug=True: Shows detailed execution steps, useful for troubleshooting
        • log_file: Redirects logs to a file instead of console output
        • Logger tracks all code execution attempts and results

        Ray Cluster Integration:
        The provided ray_cluster is used to monitor cluster state and trigger
        autoscaling when needed (especially in SLURM environments).

        Args:
            ray_cluster: Connected RayCluster instance for distributed execution
            log_file: Optional file path for log output (None = console logging)
            debug: Whether to enable verbose debug logging for troubleshooting

        Example:
            ```python
            # Basic setup with console logging
            executor = CodeExecutor(ray_cluster=my_cluster)

            # Setup with file logging and debug output
            executor = CodeExecutor(
                ray_cluster=my_cluster,
                log_file="/var/log/code_executor.log",
                debug=True
            )
            ```

        Note:
            Admin permissions must be configured via initialize() before executing code.
        """
        # Set up logging
        self.logger = create_logger(
            name="CodeExecutor",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )
        self.ray_cluster = ray_cluster
        self.admin_users = None

    def _load_func_from_source(self, code: str, function_name: str) -> Callable:
        """
        Parse Python source code and extract a specific function for execution.

        This method acts like a "function extractor" that takes a block of Python code,
        executes it safely to define all the functions and variables, then pulls out
        the specific function you want to run. It's like copying a recipe from a
        cookbook and preparing it for cooking.

        Code Processing:
        • Automatically removes common indentation (handles copy-pasted code)
        • Executes code in isolated namespace (no access to global variables)
        • Extracts the named function after successful execution
        • Returns the actual function object ready for calling

        Safety Features:
        • Code executes in controlled environment with limited scope
        • Syntax errors are caught and reported clearly
        • No side effects on the main Python environment

        Args:
            code: Python source code containing function definitions
            function_name: Exact name of the function to extract and return

        Returns:
            The extracted function object, ready to call with arguments

        Raises:
            SyntaxError: The provided code has Python syntax errors
            NameError: Code references undefined variables during execution
            Exception: Other errors during code parsing or execution

        Example:
            ```python
            code = '''
                def calculate_average(numbers):
                    return sum(numbers) / len(numbers)

                def calculate_total(numbers):
                    return sum(numbers)
            '''

            # Extract just the average function
            avg_func = executor._load_func_from_source(code, "calculate_average")
            result = avg_func([1, 2, 3, 4, 5])  # Returns 3.0
            ```

        Note:
            This is an internal method used by run_code() when processing
            source code mode execution.
        """
        exec_namespace = {}
        exec(textwrap.dedent(code), exec_namespace)
        user_func = exec_namespace.get(function_name)
        return user_func

    async def initialize(self, admin_users: List[str]) -> None:
        """
        Configure admin permissions for secure code execution access.

        This is the "security setup" step that determines who can execute arbitrary
        Python code through this service. Think of it as setting up the access control
        list for a powerful but potentially dangerous tool.

        Security Importance:
        Since this service allows execution of arbitrary Python code, it's critical
        to limit access to only trusted administrators who understand the implications.
        Anyone with access can potentially read files, modify data, or consume
        computational resources.

        Admin User Identification:
        The admin_users list should contain user identifiers that match what Hypha
        provides in the context.user.id field during service calls. These are
        typically email addresses or unique user IDs from your authentication system.

        Args:
            admin_users: List of trusted user identifiers authorized for code execution

        Example:
            ```python
            # Set up admin access for specific users
            await executor.initialize(admin_users=[
                "admin@company.com",
                "data-scientist@company.com",
                "system-admin@company.com"
            ])
            ```

            There is no wildcard form. The worker drops a '*' entry from its
            admin users before this is called, because run_code on a wildcard
            list is arbitrary code execution open to every caller that can
            reach the Hypha server, anonymous ones included.

        Note:
            This method must be called before any run_code() attempts,
            otherwise all execution requests will be rejected with PermissionError.
        """
        self.admin_users = admin_users

    @schema_method(arbitrary_types_allowed=True)
    async def run_code(
        self,
        code: Optional[str] = Field(
            None,
            description="Python source code string containing function definitions to execute. Must define a function with the specified function_name. Use triple quotes for multi-line code. Example: 'def analyze(data): return sum(data)'. Required when mode='source', ignored when mode='pickle'.",
        ),
        function_name: Optional[str] = Field(
            "analyze",
            description="Name of the Python function to execute from the provided code. The function must be defined in the code parameter. Defaults to 'analyze'. Common examples: 'process_data', 'run_analysis', 'main'. Used for logging when mode='pickle'.",
        ),
        func_bytes: Optional[bytes] = Field(
            None,
            description="Pre-serialized Python function as bytes using cloudpickle. Use this for optimized execution of complex functions. Required when mode='pickle', ignored when mode='source'. Obtain by: cloudpickle.dumps(your_function).",
        ),
        mode: Literal["source", "pickle"] = Field(
            "source",
            description="Execution mode determining input format. 'source' executes function from code string (default), 'pickle' executes pre-serialized function from func_bytes. Use 'source' for simple code execution, 'pickle' for complex functions with dependencies.",
        ),
        args: Optional[List[Any]] = Field(
            None,
            description="List of positional arguments to pass to the target function. Examples: [1, 2, 3] for numeric args, ['hello', 'world'] for string args, or [[1,2,3], {'key': 'value'}] for complex data structures. Defaults to empty list if not provided.",
        ),
        kwargs: Optional[Dict[str, Any]] = Field(
            None,
            description="Dictionary of keyword arguments to pass to the target function. Examples: {'batch_size': 32, 'learning_rate': 0.01} or {'input_file': '/path/to/data.csv', 'output_dir': '/results'}. Defaults to empty dict if not provided.",
        ),
        remote_options: Optional[Dict[str, Any]] = Field(
            None,
            description="Ray remote task configuration for resource allocation. Supported keys: 'num_cpus' (number of CPU cores), 'num_gpus' (number of GPU devices), 'memory' (bytes), 'runtime_env' (Python environment). Example: {'num_cpus': 4, 'num_gpus': 1, 'memory': 2000000000}.",
        ),
        write_stdout: Optional[Callable[[str], None]] = Field(
            None,
            description="Optional callback function for streaming stdout output in real-time. Function should accept a single string parameter. Called for each line of stdout output as it's generated during execution. Use for real-time progress monitoring.",
        ),
        write_stderr: Optional[Callable[[str], None]] = Field(
            None,
            description="Optional callback function for streaming stderr output in real-time. Function should accept a single string parameter. Called for each line of stderr output as it's generated during execution. Use for real-time error monitoring.",
        ),
        timeout: int = Field(
            180,
            description="Maximum execution time in seconds before the task is terminated. Defaults to 180 seconds (3 minutes). Increase for long-running computations, decrease for quick tasks. Range: 1-3600 seconds. Task will raise TimeoutError if exceeded.",
        ),
        context: Dict[str, Any] = Field(
            ...,
            description="Authentication context containing user information, automatically provided by Hypha during service calls.",
        ),
    ) -> Dict[str, Any]:
        """
        Execute arbitrary Python code securely in distributed Ray tasks with comprehensive resource management and monitoring.

        This method provides a secure, isolated environment for executing user-provided Python code in distributed Ray computing environments. It supports both direct source code execution and pre-serialized function execution, with configurable resource allocation, real-time output streaming, and comprehensive error handling.

        SECURITY REQUIREMENTS: This method requires admin-level permissions as it enables arbitrary Python code execution. Only authorized administrators should have access to this functionality.

        EXECUTION MODES:
        1. Source Mode (mode='source'): Execute Python code provided as a string
           - Provide code parameter with Python source code
           - Specify function_name to identify which function to execute
           - Code is parsed and executed in isolated namespace

        2. Pickle Mode (mode='pickle'): Execute pre-serialized Python functions
           - Provide func_bytes parameter with cloudpickle-serialized function
           - More efficient for complex functions with dependencies
           - Bypasses code parsing overhead

        RESOURCE ALLOCATION: Configure computational resources through remote_options:
        - num_cpus: Number of CPU cores (integer, default: 1)
        - num_gpus: Number of GPU devices (integer, default: 0)
        - memory: Memory allocation in bytes (integer, default: unlimited)
        - runtime_env: Python environment configuration (dict, default: current environment)

        REAL-TIME MONITORING: Stream execution output using callback functions:
        - write_stdout: Receives each line of standard output as it's generated
        - write_stderr: Receives each line of error output as it's generated
        - Both callbacks are called asynchronously during execution

        ERROR HANDLING: Comprehensive error capture and reporting:
        - Syntax errors in source code are caught and reported
        - Runtime exceptions include full Python traceback
        - Timeout errors for long-running executions
        - Resource allocation errors for invalid configurations

        TYPICAL WORKFLOW:
        1. Validate user has admin permissions
        2. Parse/deserialize the target function
        3. Configure Ray task with specified resources
        4. Execute function with argument passing
        5. Monitor execution with timeout protection
        6. Stream output to client callbacks
        7. Return comprehensive results with error handling

        RETURN VALUE STRUCTURE:
        Success case: {"result": function_return_value, "stdout": "...", "stderr": "..."}
        Error case: {"error": "error_message", "traceback": "...", "stdout": "...", "stderr": "..."}

        EXAMPLES:
        Basic source execution:
        await run_code(
            code="def analyze(data): return sum(data)",
            function_name="analyze",
            args=[[1,2,3,4,5]]
        )

        Resource-constrained execution:
        await run_code(
            code="def process_large_dataset(data): return expensive_computation(data)",
            function_name="process_large_dataset",
            args=[large_dataset],
            remote_options={"num_cpus": 8, "memory": 16000000000, "num_gpus": 1}
        )

        Pre-serialized function execution:
        func_bytes = cloudpickle.dumps(my_complex_function)
        await run_code(
            mode="pickle",
            func_bytes=func_bytes,
            args=[arg1, arg2],
            kwargs={"param": "value"}
        )
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"execute Python code in a Ray task",
        )
        user_id = context["user"]["id"]

        # Validate required parameters based on mode
        if mode == "pickle":
            if func_bytes is None:
                raise ValueError("func_bytes is required when mode='pickle'")
        elif mode == "source":
            if code is None:
                raise ValueError("code is required when mode='source'")
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'source' or 'pickle'")

        try:
            # Deserialize function before Ray execution
            if mode == "pickle":
                self.logger.debug(
                    f"User '{user_id}' is loading Python function '{function_name}' from pickle."
                )
                user_func = await asyncio.to_thread(cloudpickle.loads, func_bytes)
            else:
                self.logger.debug(
                    f"User '{user_id}' is loading Python function '{function_name}' from source."
                )
                user_func = await asyncio.to_thread(
                    self._load_func_from_source, code, function_name
                )

            if not callable(user_func):
                raise ValueError(
                    f"Object '{function_name}' is not callable: {user_func}"
                )

        except Exception as e:
            self.logger.error(f"Failed to load function: {e}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Configure Ray task with resource options (validates provided remote_options)
        remote_options = remote_options or {}
        try:
            configured_ray_task = ray_task.options(**remote_options)
        except ValueError as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Prepare arguments for Ray task
        args = args or []
        kwargs = kwargs or {}

        # Execute the function in a Ray task
        self.logger.info(
            f"User '{user_id}' is executing Python function '{function_name}' "
            f"in Ray task (remote_options={json.dumps(remote_options)})"
        )
        obj_ref = configured_ray_task.remote(user_func, args, kwargs)

        # If in SLURM mode, signal a resource request
        if self.ray_cluster.mode == "slurm":
            self.logger.info(
                "Notifying SLURM workers' autoscaling system of a change in cluster state."
            )
            await self.ray_cluster.monitor_cluster()

        try:
            result = await asyncio.wait_for(obj_ref, timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "error": f"Function execution timed out after {timeout} seconds.",
                "traceback": "TimeoutError: Function execution exceeded maximum allowed time.",
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Stream output to client
        if write_stdout and result.get("stdout"):
            for line in result["stdout"].splitlines():
                await write_stdout(line)
        if write_stderr and result.get("stderr"):
            for line in result["stderr"].splitlines():
                await write_stderr(line)

        return result


if __name__ == "__main__":

    class MockRayCluster:
        mode = "mock"

    executer = CodeExecutor(ray_cluster=MockRayCluster())

    # Load and execute a simple function from code
    simple_code = """
    def analyze():
        import time

        result = {
            "message": "Hello from BioEngine Worker!",
            "timestamp": time.time(),
            "python_version": "3.x"
        }

        return result
    """
    user_func = executer._load_func_from_source(simple_code, "analyze")
    assert callable(user_func)

    # Execute the function
    result = user_func()
    print(result)

    # Load and execute a parameterized function from code
    parameterized_code = """
    def analyze(name: str, multiplier: int):
    
        return {
            "greeting": f"Hello {name}!",
            "result": 42 * multiplier,
            "processed": True
        }
    """
    kwargs = {"name": "Alice", "multiplier": 3}

    user_func = executer._load_func_from_source(parameterized_code, "analyze")
    assert callable(user_func)

    # Execute the function with parameters
    result = user_func(**kwargs)
    print(result)
