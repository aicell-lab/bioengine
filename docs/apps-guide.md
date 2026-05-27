# BioEngine Applications - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Application Structure](#application-structure)
4. [Manifest Configuration](#manifest-configuration)
5. [Creating Deployments](#creating-deployments)
6. [Environment Variables and Built-in Classes](#environment-variables-and-built-in-classes)
7. [Creating Applications from Local Files](#creating-applications-from-local-files)
8. [Deployment and Usage](#deployment-and-usage)

## Introduction

BioEngine applications are flexible runtime environments that enable you to build and deploy various types of services such as:

- **Model Inference Services**: Deploy machine learning models for real-time prediction
- **Training Services**: Run distributed model training workflows
- **Data Exploration Tools**: Create interactive data analysis and visualization services
- **Custom Processing Pipelines**: Build specialized data processing workflows

BioEngine applications run on Ray Serve, providing scalable, distributed computing capabilities with automatic resource management. Applications are packaged as Hypha artifacts and can be deployed on various environments from single machines to HPC clusters.

### Key Benefits
- **Lightweight Runtime**: Minimal dependencies required (Python ≥3.11, Ray, Hypha-RPC, Pydantic, HTTPx)
- **Scalable Infrastructure**: Automatic resource allocation and cluster scaling
- **Unified Access**: WebSocket and WebRTC support for real-time communication
- **Data Integration**: Built-in access to BioEngine datasets
- **Multi-deployment Architecture**: Compose multiple services within a single application

## System Requirements

### Runtime Dependencies
BioEngine applications require only a lightweight set of dependencies:

- **Python**: ≥3.11 (must match Ray cluster version)
- **Ray**: ≥2.33.0 with client and serve components
- **Hypha-RPC**: ≥0.21.11
- **Pydantic**: ≥2.11.0  
- **HTTPx**: ≥0.28.1

> **Important**: The Python and Ray versions must be identical between the Ray cluster and your BioEngine applications.

### Additional Dependencies
Any additional Python packages not included in the lightweight runtime must be:
1. Specified in the deployment's `runtime_env` configuration
2. Imported locally within methods where they're used (not at file top level)

## Application Structure

A BioEngine application consists of a folder containing specific files that define its behavior and configuration.

### Required Files

Every BioEngine application must include:

1. **`manifest.yaml`**: Configuration file defining application metadata and deployments
2. **Python deployment file(s)**: Implementation of your application logic

### Optional Files

Recommended but not required:

- **`README.md`**: Documentation for your application
- **`tutorial.ipynb`**: Jupyter notebook demonstrating usage
- Additional Python files for multi-deployment applications

### Example Folder Structure

```
my_bioengine_app/
├── manifest.yaml           # Required: Application configuration
├── my_deployment.py        # Required: Main deployment implementation  
├── helper_deployment.py    # Optional: Additional deployment
├── README.md               # Optional: Documentation
└── tutorial.ipynb          # Optional: Usage tutorial
```

### Reference Examples

Explore these examples in the repository:
- **`tests/demo_app/`**: Simple single-deployment application
- **`tests/composition_app/`**: Multi-deployment composition example
- **`apps/model-runner/`**: Production model inference service

## Manifest Configuration

The `manifest.yaml` file defines your application's metadata, permissions, and deployment structure.

### Required Fields

```yaml
# Application identification
id: "my-app-id"                    # Unique identifier for Hypha artifact
name: "My Application Name"         # Human-readable name
description: "Brief description"    # What your application does

# Application type (always ray-serve for BioEngine)
type: ray-serve

# Access control
authorized_users:                   # List of users with access
  - "user@example.com"             # Email addresses
  - "hypha_user_id"                # Hypha user IDs  
  - "*"                            # Wildcard for all users

# Deployment configuration  
deployments:                        # List of deployments in priority order
  - "main_file:MainClass"          # Format: filename:ClassName (no .py extension)
  - "helper_file:HelperClass"      # Additional deployments (optional)
```

### Optional Fields

```yaml
# Display and metadata
id_emoji: "🚀"                     # Emoji for UI display
version: "1.0.0"                   # Application version
documentation: "README.md"         # Path to documentation file
tutorial: "tutorial.ipynb"         # Path to tutorial notebook

# Attribution
authors:
  - name: "Author Name"
    affiliation: "Organization"
    github_user: "username"
maintainers: ["username"]
license: "MIT"

# Repository and links
git_repo: "https://github.com/user/repo"
links: ["https://example.com"]
tags: ["machine-learning", "inference"]
covers: []                         # Cover images
format_version: 0.5.0              # Manifest format version
```

### Deployment List Priority

The `deployments` list order matters:
1. **First deployment**: Main entry point, automatically exposed via service APIs
2. **Subsequent deployments**: Sub-deployments, passed as parameters to main deployment

## Creating Deployments

Deployments are Ray Serve classes that define your application's behavior and resource requirements. For detailed information about Ray Serve deployments, refer to the [Ray Serve Deployment API documentation](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.Deployment.html) and the [deployment decorator parameters](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.deployment_decorator.html).

### Basic Deployment Structure

```python
import asyncio
import os
from typing import Any, Dict
from hypha_rpc.utils.schema import schema_method
from ray import serve
from pydantic import Field

@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,                    # CPU cores required
        "num_gpus": 1,                    # GPU devices (0 for CPU-only)
        "memory": 2 * 1024**3,           # Memory in bytes (2GB)
        "runtime_env": {
            "pip": ["numpy", "scikit-learn"],  # Additional dependencies
            "env_vars": {
                "CUSTOM_VAR": "value"     # Custom environment variables
            }
        }
    },
    max_ongoing_requests=10,              # Concurrent request limit
)
class MyDeployment:
    def __init__(self):
        """Initialize your deployment."""
        self.model = None

    @schema_method
    async def my_api_method(
        self, 
        input_data: str = Field(..., description="Input data to process")
    ) -> Dict[str, Any]:
        """API method exposed to users."""
        # Your application logic here
        return {"result": f"Processed: {input_data}"}
```

### Import Patterns

**Standard imports** (at file top):
```python
# Built-in Python libraries
import os
import asyncio
from typing import Any, Dict

# BioEngine runtime dependencies
from hypha_rpc.utils.schema import schema_method
from ray import serve
from pydantic import Field
```

**External dependencies** (inside methods):
```python
@schema_method
async def process_data(self, data: str) -> Dict:
    # Import external packages where needed
    import pandas as pd
    
    # Use the imported libraries
    df = pd.DataFrame(data)
    return {"shape": df.shape}
```

#### Why Import Inside Methods?

BioEngine loads your deployment script as a string, executes it in a sandbox environment, and extracts only the Ray Serve deployment class. This class is then serialized and sent to the Ray cluster for deployment.

**Key constraints:**
- **Sandbox environment**: Only packages available in the BioEngine worker environment can be imported during script loading
- **Ray cluster**: Only packages in the Ray cluster's `runtime_env` can be deserialized and used at runtime
- **Serialization**: External packages imported at the top level may fail to serialize or deserialize

**Solution**: Import additional packages inside your methods where they're actually used, and specify them in `runtime_env`:

```python
@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "runtime_env": {
            "pip": ["pandas==2.0.0", "scikit-learn==1.3.0"],
        },
    }
)
class MyDeployment:
    @schema_method
    async def analyze(self, data: list) -> Dict:
        # Import inside the method
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        df = pd.DataFrame(data)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        return {"result": scaled.tolist()}
```

### Handling Blocking Operations

Ray Serve deployments should handle requests asynchronously to maximize throughput. If you have blocking operations (I/O, CPU-intensive tasks), run them in a thread pool executor to prevent blocking other requests to the same replica:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ray import serve
from hypha_rpc.utils.schema import schema_method

@serve.deployment(
    ray_actor_options={"num_cpus": 2},
    max_ongoing_requests=10,  # Allow multiple simultaneous requests
)
class MyDeployment:
    def __init__(self):
        # Create a thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _blocking_operation(self, data):
        """A blocking CPU-intensive or I/O operation."""
        # Blocking operation here (e.g., heavy computation, file I/O)
        import time
        time.sleep(2)  # Simulated blocking work
        return f"Processed: {data}"
    
    @schema_method
    async def process(self, data: str) -> Dict:
        """Process data without blocking other requests."""
        # Run blocking operation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self._blocking_operation, 
            data
        )
        return {"result": result}
```

**When to use thread pool executors:**
- CPU-intensive computations (without releasing GIL)
- Synchronous I/O operations (file reading, database queries)
- Calls to blocking libraries without async support
- Any operation that would otherwise block the event loop

**Benefits:**
- Multiple requests can be processed simultaneously by the same replica
- Better resource utilization
- Improved throughput and response times

### Local Development and Testing

When developing and testing your deployment locally, use `DeploymentClass.func_or_class()` to get the underlying class without Ray Serve infrastructure:

```python
if __name__ == "__main__":
    import asyncio
    
    # Get the deployment class for local testing
    deployment_class = MyDeployment.func_or_class

    deployment_instance = deployment_class()
    
    # Test your methods
    result = asyncio.run(deployment_instance.process("test_data"))
    print(f"Result: {result}")
```

**Example from model-runner application:**

See [`apps/model-runner/runtime_deployment.py`](model-runner/runtime_deployment.py) for a complete example:

```python
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    
    # Reproduce BioEngine app environment
    app_workdir = Path.home() / ".bioengine" / "apps" / "model-runner"
    app_workdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(app_workdir)
    os.environ["HOME"] = str(app_workdir)
    
    # Get the deployment class for testing
    model_runner = RuntimeDeployment.func_or_class()
    
    # Test the deployment methods
    test_result = asyncio.run(
        model_runner.test(str(rdf_path), additional_requirements=["torch==2.5.1"])
    )
    print("Model testing completed successfully")
    
    # Run prediction test
    result = asyncio.run(model_runner.predict(str(rdf_path), inputs=test_image))
    print(f"Model prediction result: {result}")
```

**Benefits of local testing:**
- Fast iteration without deploying to Ray cluster
- Easy debugging with standard Python tools
- Test individual methods in isolation
- Verify logic before deployment

### Multi-Deployment Composition

For applications with multiple deployments, use `DeploymentHandle` parameters:

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class MainDeployment:
    def __init__(
        self, 
        helper_deployment: DeploymentHandle,  # Parameter name must match manifest
        model_deployment: DeploymentHandle,   # Another sub-deployment
    ):
        """Initialize with sub-deployment handles."""
        self.helper = helper_deployment
        self.model = model_deployment
        
    @schema_method
    async def process(self, data: str) -> Dict:
        """Coordinate between sub-deployments."""
        # Call helper deployment
        preprocessed = await self.helper.preprocess.remote(data)
        
        # Call model deployment  
        result = await self.model.predict.remote(preprocessed)
        
        return {"result": result}
```

**Manifest for composition:**
```yaml
deployments:
  - "main_deployment:MainDeployment"        # Main entry point
  - "helper_deployment:HelperDeployment"    # Sub-deployment
  - "model_deployment:ModelDeployment"      # Another sub-deployment
```

### Resource Configuration

#### GPU Usage
```python
@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,           # Request 1 GPU
        "num_cpus": 4,           # 4 CPU cores
        "memory": 8 * 1024**3,   # 8GB memory
    }
)
```

#### CPU-Only
```python
@serve.deployment(
    ray_actor_options={
        "num_gpus": 0,           # No GPU required
        "num_cpus": 2,           # 2 CPU cores
        "memory": 4 * 1024**3,   # 4GB memory
    }
)
```

> **Note**: GPU allocation is managed by BioEngine. The `disable_gpu=True` parameter in `deploy_app()` will override GPU requests to force CPU-only execution.

### Logging in Applications

#### Recommended Logging Approach

Use the Ray Serve logger for all logging in your BioEngine applications:

```python
import logging
from ray import serve
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from typing import Dict, Any

# Initialize logger at module level (outside the class)
# This allows use in helper functions and class methods
logger = logging.getLogger("ray.serve")

# Helper function with logging
def preprocess_data(data: str) -> str:
    """Helper function that can use the logger."""
    logger.info(f"Preprocessing data of length {len(data)}")
    processed = data.upper()
    logger.debug(f"Preprocessing complete")
    return processed

@serve.deployment(ray_actor_options={"num_cpus": 1})
class MyDeployment:
    def __init__(self):
        logger.info("Initializing MyDeployment")
        self.initialized = True
        
    @schema_method
    async def process(self, data: str = Field(..., description="Input data")) -> Dict[str, Any]:
        """Process data with detailed logging."""
        logger.info(f"Processing request with data length: {len(data)}")
        
        try:
            # Use helper function
            result = preprocess_data(data)
            logger.info("Request processed successfully")
            return {"result": result, "status": "success"}
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}
```

#### Logging Best Practices

**Log Levels:**
- **`logger.debug()`**: Detailed diagnostic information (e.g., variable values, execution flow)
- **`logger.info()`**: General informational messages (e.g., request received, processing complete)
- **`logger.warning()`**: Warning messages for potentially problematic situations
- **`logger.error()`**: Error messages for failures that don't crash the application
- **`logger.critical()`**: Critical errors that may cause application failure

**Usage Guidelines:**
```python
# Good: Informative, structured logging
logger.info(f"Model inference completed in {elapsed_time:.2f}s for batch size {batch_size}")

# Good: Include context in error logs
try:
    result = model.predict(data)
except Exception as e:
    logger.error(f"Prediction failed for input shape {data.shape}: {e}", exc_info=True)
    raise

# Avoid: Too verbose for production
logger.debug(f"Variable x = {x}, y = {y}, z = {z}")  # Use sparingly

# Avoid: Not enough context
logger.error("Error occurred")  # What error? Where?
```

#### Viewing Application Logs

Logs are accessible through the `get_app_status()` method:

```python
# Get application status with logs
status = await bioengine_worker_service.get_app_status(
    application_ids=["my-app-id"],
    logs_tail=50,           # Get last 50 log lines per replica
    n_previous_replica=1,   # Include logs from 1 previous replica
)

# Access logs for each deployment
for deployment_name, deployment_info in status["deployments"].items():
    print(f"Logs for {deployment_name}:")
    print(deployment_info["logs"])
```

**Log Parameters:**
- **`logs_tail`**: Number of most recent log lines to retrieve per replica (default: 30)
  - Set to `-1` to retrieve all available logs
  - Higher values may slow down status requests
- **`n_previous_replica`**: Number of previous (stopped) replicas to include logs from (default: 0)
  - Useful for debugging deployments that failed and restarted
  - Includes logs from replicas that are no longer running

## Environment Variables and Built-in Classes

### Available Environment Variables

Your deployment has access to these environment variables:

```python
import os

# Application environment
home_dir = os.environ["HOME"]                    # Application working directory  
tmp_dir = os.environ["TMPDIR"]                   # Temporary directory

# Hypha connection details
server_url = os.environ["HYPHA_SERVER_URL"]      # Hypha server URL
workspace = os.environ["HYPHA_WORKSPACE"]        # Current workspace
artifact_id = os.environ["HYPHA_ARTIFACT_ID"]    # Full artifact ID (workspace/name)
worker_service_id = os.environ["BIOENGINE_WORKER_SERVICE_ID"]  # Worker service ID

# Authentication (if provided via deploy_app)
token = os.environ.get("HYPHA_TOKEN")            # User authentication token
```

### Built-in Dataset Access

Every deployment automatically has access to the BioEngine datasets manager:

#### `self.bioengine_datasets`

```python
from bioengine.datasets import BioEngineDatasets

self.bioengine_datasets: BioEngineDatasets
```

Access BioEngine datasets with streaming support for zarr files:

```python
@schema_method
async def load_dataset(self, dataset_id: str) -> Dict:
    """Load data from BioEngine datasets."""
    
    # List available datasets
    datasets = await self.bioengine_datasets.list_datasets()
    dataset_info = datasets[dataset_id]
    print(f"Dataset info: {dataset_info}")
    
    # List files in a dataset
    selected_dataset = list(datasets.keys())[0]
    files = await self.bioengine_datasets.list_files(
        dataset_id=selected_dataset,
        dir_path=None,
        token=None,
    )
    for file in files:
        print(f"File: {file}")
    
    # Load a specific file
    file_content = await self.bioengine_datasets.get_file(
        dataset_id=selected_dataset, 
        file_path="example.txt", 
        token=None,
    )
    file_text = file_content.decode("utf-8")
    print(f"File content: {file_text}")

    # Load a Zarr dataset
    zarr_store = await self.bioengine_datasets.get_file(
        dataset_id=selected_dataset,
        file_path="data.zarr", 
        token=None,
    )
    zarr_group = zarr.open_group(store=zarr_store, mode="r")
    
    return {
        "dataset_info": dataset_info,  # Dataset manifest info 
        "files": files,  # List of files in dataset
        "file_content": file_text,  # Decoded text file content
        "zarr_keys": list(zarr_group.array_keys())  # Keys in Zarr group
    }
```

To list files in a specific directory within a dataset, use the `dir_path` parameter:

```python
files_in_subdir = await self.bioengine_datasets.list_files(
    dataset_id=selected_dataset,
    dir_path="subdirectory/path",  # Specify the directory path
    token=None,
)
```

Get a file within a subdirectory by providing the full relative path:

```python
file_content = await self.bioengine_datasets.get_file(
    dataset_id=selected_dataset, 
    file_path="subdirectory/path/example.txt",  # Full path to the file
    token=None,
)
```

For accessing datasets with restricted access a token is required for user authentication. `BioEngineDatasets` will default to the token that is provided during application deployment. It is also possible to provide a different token when calling the methods.

### BioEngine Lifecycle Methods

BioEngine provides optional lifecycle methods for deployment initialization and testing. The initialization order is:

1. **`__init__()`**: Synchronous initialization
2. **`async_init()`**: Asynchronous initialization (optional)
3. **`test_deployment()`**: Started as a background task (optional)

If `test_deployment()` fails, the deployment will be marked as unhealthy on the next `check_health()` call.

#### `async_init()` - Optional Async Initialization

```python
async def async_init(self) -> None:
    """Called after __init__() for async initialization."""
    # Load models, initialize connections, etc.
    self.model = await self.load_model()
    self.database = await self.connect_to_database()
```

#### `test_deployment()` - Optional Health Check

```python
async def test_deployment(self) -> None:
    """Runs once as a background task after async_init() to test deployment functionality."""
    # Verify deployment functionality
    test_input = "test"
    result = await self.model.predict(test_input)
    assert "expected_output" in result, "Deployment test failed"
```

### Ray Serve Built-in Methods

Ray Serve provides additional lifecycle methods:

#### `check_health()` - Health Check Endpoint

```python
def check_health(self) -> None:
    """
    Called by Ray Serve to verify deployment health.
    Raises an exception if the deployment is unhealthy.
    """
    # Perform health checks
    if not self.model_loaded:
        raise RuntimeError("Model not loaded")
```

This method is called periodically by Ray Serve. If it raises an exception, the replica is marked as unhealthy and may be restarted. Frequency and timeouts of health checks can be configured in Ray Serve Deployment (`@ray.serve.deployment`) parameters.

### Schema Methods

Use `@schema_method` to expose API methods:

```python
@schema_method
async def process_data(
    self,
    input_text: str = Field(..., description="Text to process"),
    max_length: int = Field(100, description="Maximum output length"),
    temperature: float = Field(0.7, description="Sampling temperature")
) -> Dict[str, Any]:
    """
    Process input text with the model.
    
    This method processes the input text and returns structured results.
    """
    # Method implementation
    return {
        "processed_text": "...",
        "confidence": 0.95,
        "metadata": {"length": len(input_text)}
    }
```

The `Field` descriptions and method docstring become part of the API documentation.

## Creating Applications from Local Files

Once you've created your application directory with the required files, you can upload it to Hypha as an artifact using either the command-line script or the web interface.

### Method 1: Using the Command-Line Script

#### Setup Requirements

Before using the `upload_app.py` script, you need to clone the BioEngine repository and install it:

```bash
# Clone the repository
git clone https://github.com/aicell-lab/bioengine.git
cd bioengine-worker

# Install BioEngine
pip install .
```

**Prepare your application directory**: Ensure your directory contains at minimum:
- `manifest.yaml` with all required fields
- Python deployment file(s) referenced in the manifest

#### How to Upload

```bash
python scripts/upload_app.py --directory <path-to-your-app>
```

This command will:
1. Read all files from the specified directory
2. Validate the `manifest.yaml` file
3. Upload the application to Hypha as an artifact
4. Display the created artifact ID (e.g., `your-workspace/your-app`)

**Using a BioEngine Worker for Upload (Recommended)**

You can use the `--worker-service-id` flag to upload applications through a BioEngine worker service. This approach has several advantages:

```bash
python scripts/upload_app.py \
    --directory <path-to-your-app> \
    --worker-service-id bioimage-io/bioengine-worker
```

**Advantages:**
- **Personal authentication**: Use your own credentials via interactive login or token
- **Workspace access**: Create/update artifacts in the worker's workspace (e.g., `bioimage-io`) if you're an authorized user
- **Permission-based**: No need for direct workspace credentials - your user permissions determine access

**Authentication**: You need to authenticate with Hypha to upload applications. You have three options:
- **Environment variable**: Export your Hypha authentication token:
    ```bash
    export HYPHA_TOKEN="your-hypha-token-here"
    ```
- **`--token` command-line argument**: You can also pass the token directly as an argument to the script:
    ```bash
    python scripts/upload_app.py --directory <path-to-your-app> --token "your-hypha-token-here"
    ```
- **Interactive login**: If no token is provided, you'll be prompted to log in interactively when running the script.

A token can be obtained from the Hypha server web interface at [hypha.aicell.io](https://hypha.aicell.io/).

In addition to authentication, you can specify the Hypha server URL and workspace using command-line arguments:
- `--server-url`: Hypha server URL (default: `https://hypha.aicell.io`)
- `--workspace`: Target workspace for the artifact (default: your default workspace)


**Full Example:**
```bash
python scripts/upload_app.py \
    --directory "apps/model-runner" \
    --server-url "https://hypha.aicell.io" \
    --workspace "bioimage-io" \
    --token "$HYPHA_TOKEN"
```

**Example with BioEngine Worker:**
```bash
python scripts/upload_app.py \
    --directory "apps/model-runner" \
    --server-url "https://hypha.aicell.io" \
    --worker-service-id "bioimage-io/bioengine-worker"
```

This will authenticate you interactively and upload the application to the `bioimage-io` workspace through the worker service.

#### What Happens During Upload

The script performs the following steps:

1. **Validation**: Checks that your manifest contains all required fields:
   - `name`, `id`, `id_emoji`, `description`, `type`, `deployments`, `authorized_users`
   - Validates that `type` is set to `ray-serve`
   - Verifies deployment and authorized_users lists are non-empty

2. **File Processing**: 
   - Reads all files in your directory (including subdirectories)
   - Text files (`.py`, `.yaml`, `.md`, etc.) are uploaded as-is
   - Binary files are automatically base64-encoded

3. **Artifact Creation**:
   - Creates or updates the artifact in the Hypha artifact manager
   - Places the artifact in the `applications` collection
   - Adds metadata like `created_by` field with your user ID

4. **File Management**: If updating an existing artifact:
   - Uploads all new files
   - Removes files that existed before but are not in the new directory
   - Only removes old files after all new files upload successfully

#### Success Output

Upon successful upload, you'll see:
```
Application created successfully!
Artifact ID: your-workspace/my-app-id
```

You can now deploy this application using the BioEngine worker service.

#### Common Issues

**Error: "Manifest is missing required field"**
- Solution: Check that your `manifest.yaml` includes all required fields listed in the [Manifest Configuration](#manifest-configuration) section

**Error: "Application type must be 'ray-serve'"**
- Solution: Ensure your manifest has `type: ray-serve`

**Error: "Invalid artifact alias"**
- Solution: The `id` field in your manifest must use lowercase letters, numbers, and hyphens only (no underscores or slashes)

**Authentication error or login prompt**
- The script will prompt you to log in interactively if `HYPHA_TOKEN` is not set
- Alternatively, you can export `HYPHA_TOKEN` as an environment variable to skip the interactive login

### Method 2: Using the Hypha Web Interface

You can also create and upload applications directly through the Hypha web interface without cloning the repository or installing BioEngine.

#### Steps

1. **Navigate to Hypha**: Go to [https://hypha.aicell.io/](https://hypha.aicell.io/)

2. **Access your workspace**:
   - Click on your profile picture in the top right corner
   - Select **"My Workspace"**

3. **Open Artifacts section**:
   - Click on **"Artifacts"** in the navigation menu

4. **Create new artifact**:
   - Click the **"+ Create Artifact"** button

5. **Configure and upload**:
   - Fill in the artifact details (name, description, etc.)
   - Set the artifact type to **"application"**
   - Upload your application files (`manifest.yaml`, Python files, etc.)
   - Click **"Create"** to finalize

The web interface provides a user-friendly way to manage your applications without needing command-line tools, and it automatically validates your manifest and files during the upload process.

## Deployment and Usage

### Deploying Applications

First, connect to the Hypha server and get the BioEngine worker service:

```python
from hypha_rpc import connect_to_server, login

# Authenticate with Hypha server
token = await login({"server_url": "https://hypha.aicell.io"})

# Connect to Hypha server
hypha_client = await connect_to_server(
    {
        "server_url": "https://hypha.aicell.io",
        "token": token,
    }
)

# Get BioEngine worker service
workspace = hypha_client.config.workspace
bioengine_worker_service_id = f"{workspace}/bioengine-worker"
bioengine_worker_service = await hypha_client.get_service(bioengine_worker_service_id)
```

Be aware that there can be multiple BioEngine worker services in a workspace. If this is the case, ensure you are using the full service ID of the desired worker, including client ID (format: "<workspace>/<client-id>:bioengine-worker").

Deploy your application using the BioEngine worker service:

```python
# Basic deployment - creates a new application with auto-generated ID
application_id = await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app"
)

# Deployment with custom application ID
application_id = await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app",
    application_id="my-custom-id",      # Specify your own ID
)

# Advanced deployment with full configuration
application_id = await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app",
    version=None,                       # Latest version
    application_id="custom-id",         # Custom instance ID
    hypha_token="your_token_here",      # User authentication
    disable_gpu=False,                  # Enable GPU usage
    max_ongoing_requests=20,            # Concurrent request limit
    application_kwargs={                # Per-deployment initialization args
        "MainDeployment": {
            "model_path": "/path/to/model"
        }
    },
    application_env_vars={              # Per-deployment environment variables
        "MainDeployment": {
            "DEBUG_MODE": "true"
        }
    }
)
```

#### Updating Existing Applications

You can update a running application by calling `deploy_app` with the **same `application_id`** but a **different `artifact_id`** (or version). This allows you to:
- Deploy a newer version of your application
- Switch to a completely different artifact while keeping the same application ID
- Update configuration parameters while keeping others unchanged

**How Updates Work:**
- **Application ID**: Must match the existing application you want to update (cannot be changed)
- **Artifact ID**: Can be changed to deploy from a different artifact
- **Parameters**: Any unspecified parameters are inherited from the currently running application
- **Specified parameters**: Override the current values
- **Timestamps**: Original creation time (`started_at`) is preserved, `last_updated_at` is updated

```python
# Initial deployment
app_id = await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app-v1",
    application_id="my-app",
    disable_gpu=False,
    max_ongoing_requests=10,
)

# Update to a newer artifact version - keeps all other settings
await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app-v2",  # New artifact
    application_id="my-app",             # Same app ID = update
    # All other parameters inherited from current deployment
)

# Update specific parameters only - keeps artifact and other settings
await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app-v2",  # Can keep same or change
    application_id="my-app",             # Same app ID = update
    max_ongoing_requests=20,             # Update this parameter
    # Other parameters (disable_gpu, etc.) inherited
)

# Update with new configuration
await bioengine_worker_service.deploy_app(
    artifact_id="workspace/my-app-v2",
    application_id="my-app",
    application_kwargs={                 # New initialization parameters
        "MainDeployment": {
            "model_path": "/new/path/to/model"
        }
    },
    hypha_token="new_token",            # Updated authentication
    # version, disable_gpu, etc. inherited if not specified
)
```

**Update Behavior:**
- The existing deployment is gracefully stopped
- A new deployment is created with the updated configuration
- Original creation timestamp is preserved
- Update timestamp and user are recorded
- Service IDs may change as replicas are recreated

**When to Update vs. Redeploy:**
- **Update** (same `application_id`): When you want to maintain the same logical application but with updated code/config
- **New deployment** (different `application_id`): When you want to run multiple versions simultaneously or create a completely separate instance

### Deploy Application Parameters

The `deploy_app` method supports these parameters:

- **`artifact_id`** *(required)*: Application artifact identifier
- **`version`** *(optional)*: Specific artifact version to deploy
- **`application_id`** *(optional)*: Custom instance ID for deployment
- **`application_kwargs`** *(optional)*: Initialization parameters per deployment
- **`application_env_vars`** *(optional)*: Environment variables per deployment  
- **`hypha_token`** *(optional)*: Authentication token for user permissions. Use this to run the application with specific user credentials, which determines access to datasets and other resources. Can be obtained from `await login()` or from the [Hypha dashboard](https://hypha.aicell.io/) (Login → Profile Picture → My Workspace → Development tab → Generate Token).
- **`disable_gpu`** *(optional)*: Force CPU-only execution (default: False)
- **`max_ongoing_requests`** *(optional)*: Concurrent request limit (default: 10)

#### Secret Environment Variables

Environment variables that start with an underscore (`_`) are treated as secrets:

```python
application_env_vars={
    "MainDeployment": {
        "API_URL": "https://api.example.com",  # Normal env var - visible in status
        "_API_KEY": "secret-key-value",        # Secret env var - hidden in status
        "_DATABASE_PASSWORD": "db-password",   # Secret env var - hidden in status
    }
}
```

**Key differences:**

- **Normal variables**: Visible in application status and logs
- **Secret variables** (prefixed with `_`): 
  - Values are hidden in application status (shown as `*****`)
  - Values are hidden in Ray actor configuration logs
  - The underscore prefix is automatically removed when set as environment variables
  - Access in code without the underscore: `os.environ["API_KEY"]` not `os.environ["_API_KEY"]`

**Example usage in deployment code:**

```python
import os

@serve.deployment(ray_actor_options={"num_cpus": 1})
class MyDeployment:
    def __init__(self):
        # Access normal env var
        self.api_url = os.environ["API_URL"]
        
        # Access secret env var (without underscore prefix)
        self.api_key = os.environ["API_KEY"]  # Not "_API_KEY"
        self.db_password = os.environ["DATABASE_PASSWORD"]  # Not "_DATABASE_PASSWORD"
```

### Accessing Deployed Applications

#### Service Discovery

```python
# Get worker status to find application services
app_id = "my-application-id"  # Replace with your application ID
app_status = await bioengine_worker_service.get_app_status(
    application_ids=[app_id]
)

# Extract service IDs
websocket_service_id = app_status["service_ids"]["websocket_service_id"]
webrtc_service_id = app_status["service_ids"]["webrtc_service_id"]
```

#### WebSocket Connection

Websocket connections send and receive their data through the connected Hypha server.

```python
# Connect via WebSocket for real-time communication
websocket_service = await hypha_client.get_service(websocket_service_id)

# Call application methods
result = await websocket_service.process_data(
    input_text="Hello BioEngine!",
    max_length=200
)
```

#### WebRTC Connection

WebRTC connections enable peer-to-peer communication for low-latency applications, larger data transfers, and privacy-sensitive use cases.

```python
# Requires aiortc to be installed
from hypha_rpc import get_rtc_service

# Connect via WebRTC for peer-to-peer communication
peer_connection = await get_rtc_service(hypha_client, webrtc_service_id)

webrtc_service = await peer_connection.get_service(app_id)

# Use WebRTC for low-latency real-time applications
# Call application methods
result = await webrtc_service.process_data(
    input_text="Hello BioEngine!",
    max_length=200
)
```

### Application Lifecycle Management

```python
# Start application
app_id = await bioengine_worker_service.deploy_app("workspace/my-app")

# Monitor application status (basic)
app_status = await bioengine_worker_service.get_app_status(
    application_ids=[app_id]
)

# Monitor with detailed logs
app_status = await bioengine_worker_service.get_app_status(
    application_ids=[app_id],
    logs_tail=100,           # Get last 100 log lines per replica
    n_previous_replica=2,    # Include logs from 2 previous replicas
)

# Access deployment logs
for deployment_name, deployment_info in app_status["deployments"].items():
    print(f"Status: {deployment_info['status']}")
    print(f"Message: {deployment_info['message']}")
    print(f"Logs:\n{deployment_info['logs']}")

# Stop application when done
await bioengine_worker_service.stop_app(app_id)
```

#### Monitoring Application Status

The `get_app_status()` method provides comprehensive information about your deployed applications:

**Parameters:**
- **`application_ids`** *(optional)*: List of application IDs to check. If `None`, returns all deployed applications. If a single-item list, returns status directly (not wrapped in a dictionary).
- **`logs_tail`** *(optional, default: 30)*: Number of recent log lines to retrieve per replica. Set to `-1` for all logs.
- **`n_previous_replica`** *(optional, default: 0)*: Number of previous (stopped/failed) replicas to include logs from.

**Returned Information:**
- Application metadata (name, description, artifact details)
- Current status (RUNNING, DEPLOYING, UNHEALTHY, etc.)
- Resource allocation and usage
- Service IDs for WebSocket/WebRTC connections
- Deployment details and replica states
- **Logs from active and previous replicas**

**Example: Debugging a Failed Deployment**

```python
# Get detailed status with extended logs
status = await bioengine_worker_service.get_app_status(
    application_ids=["my-failing-app"],
    logs_tail=-1,            # Get all available logs
    n_previous_replica=3,    # Check logs from last 3 replica attempts
)

# Check deployment status
if status["status"] == "UNHEALTHY":
    print(f"Application unhealthy: {status['message']}")
    
    # Examine logs for errors
    for deployment_name, deployment_info in status["deployments"].items():
        print(f"\nDeployment: {deployment_name}")
        print(f"Status: {deployment_info['status']}")
        print(f"Replica states: {deployment_info['replica_states']}")
        
        # Print logs to identify the issue
        if deployment_info["logs"]:
            print("\nLogs:")
            print(deployment_info["logs"])
```

---

## BioEngine Worker Setup and Usage

For information about setting up and running the BioEngine Worker itself, please refer to the [main README.md](../README.md) in the repository for complete setup instructions, including Docker and Apptainer deployment options.
