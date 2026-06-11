"""BioEngine application lifecycle: ``AppsManager`` (worker-side orchestrator),
``AppBuilder`` (build a Ray Serve application from a Hypha artifact), and
``ProxyDeployment`` (the actor that terminates the WebSocket/WebRTC client
bridge and forwards calls to user deployments).

This package's ``__init__`` deliberately re-exports nothing. The Ray actor that
runs ``bioengine._app.bootstrap.build_and_run_application`` imports a single
submodule (``proxy_deployment``) and must not transitively pull in
``manager.py`` — that module imports ``haikunator`` and other worker-only
dependencies that are not present in the actor's runtime_env, and a top-level
re-export here would crash unpickling on every external-cluster deploy.

Worker-side code that wants the manager should ``from bioengine.apps.manager
import AppsManager`` directly.
"""
