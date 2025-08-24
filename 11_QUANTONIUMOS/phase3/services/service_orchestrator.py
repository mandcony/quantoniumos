"""
QuantoniumOS Phase 3: Service Orchestration
Advanced orchestration system for managing quantum and classical services
"""

import asyncio
import json
import logging
import threading
import time
import uuid
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class ServiceStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    SUSPENDED = "suspended"


class ServiceType(Enum):
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"
    BRIDGE = "bridge"
    API = "api"
    GUI = "gui"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ServiceDependency:
    """Represents a dependency between services"""

    service_name: str
    dependency_type: str = "hard"  # "hard", "soft", "optional"
    min_version: str = "1.0.0"
    startup_delay: float = 0.0


@dataclass
class ServiceConfiguration:
    """Configuration for a service"""

    name: str
    service_type: ServiceType
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[ServiceDependency] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 5.0
    health_check_interval: float = 30.0
    timeout_start: float = 60.0
    timeout_stop: float = 30.0


@dataclass
class ServiceInstance:
    """Represents a running service instance"""

    config: ServiceConfiguration
    status: ServiceStatus = ServiceStatus.STOPPED
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    restart_count: int = 0
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ServiceOrchestrator:
    """
    Advanced service orchestration system for QuantoniumOS
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Service registry
        self.services: Dict[str, ServiceInstance] = {}
        self.service_handlers: Dict[str, Callable] = {}

        # Dependency graph
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_dependency_graph: Dict[str, List[str]] = defaultdict(list)

        # Event system
        self.event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue()

        # Orchestrator state
        self.is_running = False
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []

        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.event_processor_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.orchestrator_metrics = {
            "services_started": 0,
            "services_stopped": 0,
            "services_restarted": 0,
            "total_uptime": 0.0,
            "health_checks_performed": 0,
            "events_processed": 0,
        }

        self.logger.info("Service Orchestrator initialized")

    def register_service(self, config: ServiceConfiguration, handler: Callable) -> bool:
        """Register a new service with the orchestrator"""
        try:
            if config.name in self.services:
                self.logger.warning(f"Service {config.name} already registered")
                return False

            # Create service instance
            instance = ServiceInstance(config=config)
            self.services[config.name] = instance
            self.service_handlers[config.name] = handler

            # Build dependency graph
            self._update_dependency_graph(config)

            # Validate dependencies
            if not self._validate_dependencies(config.name):
                self.logger.error(f"Invalid dependencies for service {config.name}")
                del self.services[config.name]
                del self.service_handlers[config.name]
                return False

            self.logger.info(
                f"Registered service: {config.name} (type: {config.service_type.value})"
            )

            # Emit registration event
            asyncio.create_task(
                self._emit_event(
                    "service_registered",
                    {
                        "service_name": config.name,
                        "service_type": config.service_type.value,
                        "version": config.version,
                    },
                )
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to register service {config.name}: {e}")
            return False

    def unregister_service(self, service_name: str) -> bool:
        """Unregister a service from the orchestrator"""
        try:
            if service_name not in self.services:
                self.logger.warning(f"Service {service_name} not registered")
                return False

            # Stop service if running
            if self.services[service_name].status == ServiceStatus.RUNNING:
                asyncio.create_task(self.stop_service(service_name))

            # Remove from registry
            del self.services[service_name]
            del self.service_handlers[service_name]

            # Update dependency graphs
            self._remove_from_dependency_graph(service_name)

            self.logger.info(f"Unregistered service: {service_name}")

            # Emit unregistration event
            asyncio.create_task(
                self._emit_event("service_unregistered", {"service_name": service_name})
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister service {service_name}: {e}")
            return False

    async def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name not in self.services:
            self.logger.error(f"Service {service_name} not registered")
            return False

        instance = self.services[service_name]
        config = instance.config

        if instance.status == ServiceStatus.RUNNING:
            self.logger.info(f"Service {service_name} is already running")
            return True

        try:
            self.logger.info(f"Starting service: {service_name}")
            instance.status = ServiceStatus.STARTING

            # Start dependencies first
            for dep in config.dependencies:
                if dep.dependency_type == "hard":
                    if not await self.start_service(dep.service_name):
                        self.logger.error(
                            f"Failed to start dependency {dep.service_name}"
                        )
                        instance.status = ServiceStatus.ERROR
                        instance.error_message = (
                            f"Dependency {dep.service_name} failed to start"
                        )
                        return False

                    # Wait for startup delay
                    if dep.startup_delay > 0:
                        await asyncio.sleep(dep.startup_delay)

            # Call service handler
            handler = self.service_handlers[service_name]

            # Start service with timeout
            try:
                start_task = asyncio.create_task(handler("start", config.config_params))
                await asyncio.wait_for(start_task, timeout=config.timeout_start)

                # Service started successfully
                instance.status = ServiceStatus.RUNNING
                instance.start_time = datetime.now()
                instance.restart_count = 0
                instance.error_message = None

                self.orchestrator_metrics["services_started"] += 1

                self.logger.info(f"Service {service_name} started successfully")

                # Emit start event
                await self._emit_event(
                    "service_started",
                    {
                        "service_name": service_name,
                        "start_time": instance.start_time.isoformat(),
                    },
                )

                return True

            except asyncio.TimeoutError:
                self.logger.error(f"Service {service_name} start timeout")
                instance.status = ServiceStatus.ERROR
                instance.error_message = "Start timeout"
                return False
            except Exception as e:
                self.logger.error(f"Service {service_name} start failed: {e}")
                instance.status = ServiceStatus.ERROR
                instance.error_message = str(e)
                return False

        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            instance.status = ServiceStatus.ERROR
            instance.error_message = str(e)
            return False

    async def stop_service(self, service_name: str, force: bool = False) -> bool:
        """Stop a specific service"""
        if service_name not in self.services:
            self.logger.error(f"Service {service_name} not registered")
            return False

        instance = self.services[service_name]

        if instance.status == ServiceStatus.STOPPED:
            self.logger.info(f"Service {service_name} is already stopped")
            return True

        try:
            self.logger.info(f"Stopping service: {service_name}")
            instance.status = ServiceStatus.STOPPING

            # Stop dependent services first (if not forced)
            if not force:
                dependents = self.reverse_dependency_graph.get(service_name, [])
                for dependent in dependents:
                    if self.services[dependent].status == ServiceStatus.RUNNING:
                        await self.stop_service(dependent)

            # Call service handler
            handler = self.service_handlers[service_name]

            try:
                stop_task = asyncio.create_task(
                    handler("stop", instance.config.config_params)
                )
                await asyncio.wait_for(stop_task, timeout=instance.config.timeout_stop)

                # Service stopped successfully
                instance.status = ServiceStatus.STOPPED
                instance.stop_time = datetime.now()
                instance.error_message = None

                self.orchestrator_metrics["services_stopped"] += 1

                self.logger.info(f"Service {service_name} stopped successfully")

                # Emit stop event
                await self._emit_event(
                    "service_stopped",
                    {
                        "service_name": service_name,
                        "stop_time": instance.stop_time.isoformat(),
                    },
                )

                return True

            except asyncio.TimeoutError:
                self.logger.error(f"Service {service_name} stop timeout")
                if force:
                    instance.status = ServiceStatus.STOPPED
                    return True
                else:
                    instance.status = ServiceStatus.ERROR
                    instance.error_message = "Stop timeout"
                    return False
            except Exception as e:
                self.logger.error(f"Service {service_name} stop failed: {e}")
                instance.status = ServiceStatus.ERROR
                instance.error_message = str(e)
                return False

        except Exception as e:
            self.logger.error(f"Failed to stop service {service_name}: {e}")
            instance.status = ServiceStatus.ERROR
            instance.error_message = str(e)
            return False

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        instance = self.services.get(service_name)
        if not instance:
            self.logger.error(f"Service {service_name} not registered")
            return False

        self.logger.info(f"Restarting service: {service_name}")

        # Stop the service
        if instance.status == ServiceStatus.RUNNING:
            if not await self.stop_service(service_name):
                self.logger.error(f"Failed to stop service {service_name} for restart")
                return False

        # Wait for restart delay
        if instance.config.restart_delay > 0:
            await asyncio.sleep(instance.config.restart_delay)

        # Start the service
        if await self.start_service(service_name):
            instance.restart_count += 1
            self.orchestrator_metrics["services_restarted"] += 1

            await self._emit_event(
                "service_restarted",
                {"service_name": service_name, "restart_count": instance.restart_count},
            )

            return True
        else:
            return False

    async def start_all_services(self) -> bool:
        """Start all registered services in dependency order"""
        self.logger.info("Starting all services")

        try:
            # Calculate startup order
            startup_order = self._calculate_startup_order()

            success_count = 0
            for service_name in startup_order:
                if await self.start_service(service_name):
                    success_count += 1
                else:
                    self.logger.error(f"Failed to start service {service_name}")

            self.logger.info(f"Started {success_count}/{len(startup_order)} services")
            return success_count == len(startup_order)

        except Exception as e:
            self.logger.error(f"Failed to start all services: {e}")
            return False

    async def stop_all_services(self) -> bool:
        """Stop all running services in reverse dependency order"""
        self.logger.info("Stopping all services")

        try:
            # Calculate shutdown order (reverse of startup)
            shutdown_order = self._calculate_shutdown_order()

            success_count = 0
            for service_name in shutdown_order:
                if self.services[service_name].status == ServiceStatus.RUNNING:
                    if await self.stop_service(service_name):
                        success_count += 1
                    else:
                        self.logger.error(f"Failed to stop service {service_name}")
                else:
                    success_count += 1  # Already stopped

            self.logger.info(f"Stopped {success_count}/{len(shutdown_order)} services")
            return success_count == len(shutdown_order)

        except Exception as e:
            self.logger.error(f"Failed to stop all services: {e}")
            return False

    async def start_orchestrator(self) -> bool:
        """Start the orchestrator itself"""
        if self.is_running:
            self.logger.info("Orchestrator is already running")
            return True

        try:
            self.logger.info("Starting Service Orchestrator")
            self.is_running = True

            # Start background tasks
            self.monitor_task = asyncio.create_task(self._service_monitor())
            self.event_processor_task = asyncio.create_task(self._event_processor())

            # Start all services
            await self.start_all_services()

            self.logger.info("Service Orchestrator started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {e}")
            self.is_running = False
            return False

    async def stop_orchestrator(self) -> bool:
        """Stop the orchestrator"""
        if not self.is_running:
            self.logger.info("Orchestrator is already stopped")
            return True

        try:
            self.logger.info("Stopping Service Orchestrator")

            # Stop all services
            await self.stop_all_services()

            # Stop background tasks
            self.is_running = False

            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            if self.event_processor_task:
                self.event_processor_task.cancel()
                try:
                    await self.event_processor_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Service Orchestrator stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop orchestrator: {e}")
            return False

    def get_service_status(self, service_name: str = None) -> Dict[str, Any]:
        """Get status of specific service or all services"""
        if service_name:
            if service_name in self.services:
                instance = self.services[service_name]
                return {
                    "name": instance.config.name,
                    "type": instance.config.service_type.value,
                    "version": instance.config.version,
                    "status": instance.status.value,
                    "start_time": instance.start_time.isoformat()
                    if instance.start_time
                    else None,
                    "uptime": str(datetime.now() - instance.start_time)
                    if instance.start_time
                    else None,
                    "restart_count": instance.restart_count,
                    "health_status": instance.health_status,
                    "error_message": instance.error_message,
                    "metrics": instance.metrics,
                }
            else:
                return {"error": f"Service {service_name} not found"}
        else:
            return {
                "orchestrator_running": self.is_running,
                "total_services": len(self.services),
                "running_services": len(
                    [
                        s
                        for s in self.services.values()
                        if s.status == ServiceStatus.RUNNING
                    ]
                ),
                "services": {
                    name: {
                        "status": instance.status.value,
                        "type": instance.config.service_type.value,
                        "restart_count": instance.restart_count,
                        "health_status": instance.health_status,
                    }
                    for name, instance in self.services.items()
                },
                "metrics": self.orchestrator_metrics,
            }

    def add_event_listener(self, event_type: str, callback: Callable):
        """Add an event listener"""
        self.event_listeners[event_type].append(callback)
        self.logger.debug(f"Added event listener for {event_type}")

    def remove_event_listener(self, event_type: str, callback: Callable):
        """Remove an event listener"""
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
            self.logger.debug(f"Removed event listener for {event_type}")

    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event to the event queue"""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4()),
        }
        await self.event_queue.put(event)

    async def _event_processor(self):
        """Background task to process events"""
        while self.is_running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                # Call all listeners for this event type
                listeners = self.event_listeners.get(event["type"], [])
                for listener in listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(event)
                        else:
                            listener(event)
                    except Exception as e:
                        self.logger.error(f"Event listener error: {e}")

                self.orchestrator_metrics["events_processed"] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Event processor error: {e}")

    async def _service_monitor(self):
        """Background task to monitor service health"""
        while self.is_running:
            try:
                for service_name, instance in self.services.items():
                    if instance.status == ServiceStatus.RUNNING:
                        await self._health_check_service(service_name)

                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Service monitor error: {e}")

    async def _health_check_service(self, service_name: str):
        """Perform health check on a service"""
        instance = self.services[service_name]
        config = instance.config

        # Check if health check is due
        now = datetime.now()
        if (
            instance.last_health_check
            and (now - instance.last_health_check).total_seconds()
            < config.health_check_interval
        ):
            return

        try:
            # Call service handler for health check
            handler = self.service_handlers[service_name]
            health_result = await handler("health_check", config.config_params)

            if health_result:
                instance.health_status = "healthy"
            else:
                instance.health_status = "unhealthy"

                # Auto-restart if configured
                if (
                    config.auto_restart
                    and instance.restart_count < config.max_restart_attempts
                ):
                    self.logger.warning(
                        f"Service {service_name} unhealthy, attempting restart"
                    )
                    await self.restart_service(service_name)

            instance.last_health_check = now
            self.orchestrator_metrics["health_checks_performed"] += 1

        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            instance.health_status = "check_failed"

    def _update_dependency_graph(self, config: ServiceConfiguration):
        """Update dependency graphs with new service"""
        service_name = config.name

        # Clear existing dependencies
        self.dependency_graph[service_name] = []

        # Add dependencies
        for dep in config.dependencies:
            self.dependency_graph[service_name].append(dep.service_name)
            self.reverse_dependency_graph[dep.service_name].append(service_name)

    def _remove_from_dependency_graph(self, service_name: str):
        """Remove service from dependency graphs"""
        # Remove from dependency graph
        if service_name in self.dependency_graph:
            for dep in self.dependency_graph[service_name]:
                if service_name in self.reverse_dependency_graph[dep]:
                    self.reverse_dependency_graph[dep].remove(service_name)
            del self.dependency_graph[service_name]

        # Remove from reverse dependency graph
        if service_name in self.reverse_dependency_graph:
            for dependent in self.reverse_dependency_graph[service_name]:
                if service_name in self.dependency_graph[dependent]:
                    self.dependency_graph[dependent].remove(service_name)
            del self.reverse_dependency_graph[service_name]

    def _validate_dependencies(self, service_name: str) -> bool:
        """Validate that all dependencies exist"""
        for dep_name in self.dependency_graph[service_name]:
            if dep_name not in self.services:
                self.logger.error(
                    f"Dependency {dep_name} not registered for service {service_name}"
                )
                return False
        return True

    def _calculate_startup_order(self) -> List[str]:
        """Calculate the order to start services based on dependencies"""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []

        def visit(service_name):
            if service_name in temp_visited:
                raise ValueError("Circular dependency detected")
            if service_name in visited:
                return

            temp_visited.add(service_name)

            for dep_name in self.dependency_graph.get(service_name, []):
                visit(dep_name)

            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)

        for service_name in self.services:
            if service_name not in visited:
                visit(service_name)

        return order

    def _calculate_shutdown_order(self) -> List[str]:
        """Calculate the order to shutdown services (reverse of startup)"""
        startup_order = self._calculate_startup_order()
        return list(reversed(startup_order))


# Global orchestrator instance
service_orchestrator = ServiceOrchestrator()


# Example service handler
async def example_service_handler(action: str, params: Dict[str, Any]) -> Any:
    """Example service handler"""
    if action == "start":
        # Service startup logic
        await asyncio.sleep(0.1)  # Simulate startup time
        return True
    elif action == "stop":
        # Service shutdown logic
        await asyncio.sleep(0.1)  # Simulate shutdown time
        return True
    elif action == "health_check":
        # Health check logic
        return True
    else:
        return False


# Example usage
if __name__ == "__main__":

    async def demo_orchestrator():
        print("QuantoniumOS Service Orchestrator Demo")
        print("=" * 50)

        # Register example services
        quantum_service = ServiceConfiguration(
            name="quantum_engine",
            service_type=ServiceType.QUANTUM,
            version="1.0.0",
            description="Quantum computation engine",
            dependencies=[],
            config_params={"num_qubits": 100},
        )

        bridge_service = ServiceConfiguration(
            name="quantum_bridge",
            service_type=ServiceType.BRIDGE,
            version="1.0.0",
            description="Quantum-classical bridge",
            dependencies=[ServiceDependency("quantum_engine")],
            config_params={"bridge_mode": "hybrid"},
        )

        api_service = ServiceConfiguration(
            name="quantum_api",
            service_type=ServiceType.API,
            version="1.0.0",
            description="QuantoniumOS API",
            dependencies=[
                ServiceDependency("quantum_engine"),
                ServiceDependency("quantum_bridge"),
            ],
            config_params={"port": 8000},
        )

        # Register services
        service_orchestrator.register_service(quantum_service, example_service_handler)
        service_orchestrator.register_service(bridge_service, example_service_handler)
        service_orchestrator.register_service(api_service, example_service_handler)

        # Start orchestrator
        await service_orchestrator.start_orchestrator()

        # Show status
        status = service_orchestrator.get_service_status()
        print(f"Orchestrator Status:")
        print(f"Running: {status['orchestrator_running']}")
        print(f"Total Services: {status['total_services']}")
        print(f"Running Services: {status['running_services']}")

        # Wait a bit
        await asyncio.sleep(2)

        # Stop orchestrator
        await service_orchestrator.stop_orchestrator()

        print("Demo completed")

    asyncio.run(demo_orchestrator())
