# agents/enhanced_agent_3.py

from pathlib import Path
import sys
import os
import subprocess
import json
import time
import shutil
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from dataclasses import dataclass
from enum import Enum

from config import ARTIFACTS_DIR

class Platform(Enum):
    MOBILE = "mobile"
    WEB = "web"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExecutionConfig:
    platform: Platform
    use_enhanced_agent: bool = True
    enable_monitoring: bool = True
    enable_reporting: bool = True
    timeout_seconds: int = 300
    retry_attempts: int = 1

# Enhanced Execution Intelligence System
class ExecutionIntelligence:
    """Smart execution system that learns from previous runs and optimizes future executions"""
    
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.common_issues: Dict[str, int] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        self.mobile_specific_insights: Dict[str, Any] = {
            'captcha_success_rate': [],
            'gesture_optimization': {},
            'device_performance': {},
            'element_detection_patterns': {}
        }
        self.platform_insights: Dict[Platform, Dict[str, Any]] = {
            Platform.MOBILE: {
                'average_execution_time': [],
                'common_failures': {},
                'optimization_opportunities': [],
                'gesture_success_rates': {}
            },
            Platform.WEB: {
                'average_execution_time': [],
                'common_failures': {},
                'optimization_opportunities': {},
                'stealth_effectiveness': {}
            }
        }
    
    def record_execution(self, platform: Platform, execution_data: Dict[str, Any]):
        """Record execution for learning"""
        timestamp = time.time()
        
        execution_record = {
            'timestamp': timestamp,
            'platform': platform.value,
            'duration': execution_data.get('duration', 0),
            'status': execution_data.get('status', 'unknown'),
            'step_count': execution_data.get('step_count', 0),
            'errors': execution_data.get('errors', []),
            'performance_metrics': execution_data.get('performance_metrics', {}),
            'device_info': execution_data.get('device_info', {}),
            'optimization_applied': execution_data.get('optimization_applied', [])
        }
        
        self.execution_history.append(execution_record)
        
        # Update platform-specific insights
        platform_data = self.platform_insights[platform]
        platform_data['average_execution_time'].append(execution_data.get('duration', 0))
        
        # Track common failures
        for error in execution_data.get('errors', []):
            error_type = error.get('type', 'unknown')
            platform_data['common_failures'][error_type] = platform_data['common_failures'].get(error_type, 0) + 1
        
        # Mobile-specific tracking
        if platform == Platform.MOBILE:
            self._update_mobile_insights(execution_data)
        
        # Keep only last 100 executions for performance
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _update_mobile_insights(self, execution_data: Dict[str, Any]):
        """Update mobile-specific intelligence"""
        mobile_insights = self.mobile_specific_insights
        
        # Track CAPTCHA handling success
        captcha_success = execution_data.get('captcha_handled', False)
        mobile_insights['captcha_success_rate'].append(1.0 if captcha_success else 0.0)
        
        # Track gesture performance
        gestures = execution_data.get('gestures_used', [])
        for gesture in gestures:
            gesture_type = gesture.get('type', 'unknown')
            success = gesture.get('success', False)
            
            if gesture_type not in mobile_insights['gesture_optimization']:
                mobile_insights['gesture_optimization'][gesture_type] = {'success': 0, 'total': 0}
            
            mobile_insights['gesture_optimization'][gesture_type]['total'] += 1
            if success:
                mobile_insights['gesture_optimization'][gesture_type]['success'] += 1
        
        # Track device performance
        device_id = execution_data.get('device_info', {}).get('device_id', 'unknown')
        if device_id not in mobile_insights['device_performance']:
            mobile_insights['device_performance'][device_id] = []
        
        mobile_insights['device_performance'][device_id].append({
            'duration': execution_data.get('duration', 0),
            'success': execution_data.get('status') == 'completed',
            'timestamp': time.time()
        })
    
    def get_recommendations(self, platform: Platform, complexity_score: float = 0.5) -> List[str]:
        """Get intelligent recommendations based on historical data"""
        recommendations = []
        
        platform_data = self.platform_insights[platform]
        
        # Performance recommendations
        if platform_data['average_execution_time']:
            avg_time = sum(platform_data['average_execution_time']) / len(platform_data['average_execution_time'])
            if avg_time > 180:  # Over 3 minutes
                recommendations.append("Consider optimizing element waiting strategies - average execution time is high")
                recommendations.append("Enable parallel element detection for better performance")
        
        # Failure pattern recommendations
        common_failures = platform_data['common_failures']
        if common_failures:
            most_common = max(common_failures.items(), key=lambda x: x[1])
            if most_common[1] > 3:  # More than 3 occurrences
                recommendations.append(f"Implement enhanced handling for '{most_common[0]}' errors (occurred {most_common[1]} times)")
        
        # Platform-specific recommendations
        if platform == Platform.MOBILE:
            recommendations.extend(self._get_mobile_recommendations(complexity_score))
        else:
            recommendations.extend(self._get_web_recommendations(complexity_score))
        
        # Complexity-based recommendations
        if complexity_score > 0.8:
            recommendations.append("High complexity detected - enable comprehensive error handling")
            recommendations.append("Consider breaking automation into smaller, more reliable chunks")
            recommendations.append("Implement checkpoint-based recovery for long workflows")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_mobile_recommendations(self, complexity_score: float) -> List[str]:
        """Mobile-specific recommendations"""
        recommendations = []
        mobile_insights = self.mobile_specific_insights
        
        # CAPTCHA recommendations
        if mobile_insights['captcha_success_rate']:
            captcha_rate = sum(mobile_insights['captcha_success_rate']) / len(mobile_insights['captcha_success_rate'])
            if captcha_rate < 0.7:
                recommendations.append("CAPTCHA success rate is low - enable extended long press duration (20+ seconds)")
                recommendations.append("Add coordinate-based fallback for CAPTCHA handling")
        
        # Gesture recommendations
        for gesture_type, data in mobile_insights['gesture_optimization'].items():
            if data['total'] >= 5:  # Enough data points
                success_rate = data['success'] / data['total']
                if success_rate < 0.8:
                    recommendations.append(f"Optimize {gesture_type} gestures - current success rate: {success_rate:.1%}")
        
        # Device performance recommendations
        device_performances = list(mobile_insights['device_performance'].values())
        if device_performances:
            # Check if device performance is degrading
            recent_performance = device_performances[0][-5:] if device_performances[0] else []
            if recent_performance and len(recent_performance) >= 3:
                recent_avg = sum(p['duration'] for p in recent_performance) / len(recent_performance)
                if recent_avg > 120:  # Over 2 minutes
                    recommendations.append("Device performance degrading - consider device restart between sessions")
        
        return recommendations
    
    def _get_web_recommendations(self, complexity_score: float) -> List[str]:
        """Web-specific recommendations"""
        recommendations = []
        web_data = self.platform_insights[Platform.WEB]
        
        # Add web-specific recommendations based on stealth effectiveness, etc.
        if 'stealth_detection' in web_data['common_failures']:
            recommendations.append("Stealth detection failures - enhance browser fingerprinting protection")
        
        return recommendations
    
    def get_optimization_config(self, platform: Platform) -> Dict[str, Any]:
        """Get optimized configuration based on learning"""
        base_config = {
            'retry_attempts': 3,
            'element_timeout': 15,
            'action_delay': 0.5,
            'screenshot_on_failure': True
        }
        
        platform_data = self.platform_insights[platform]
        
        # Adjust based on failure patterns
        if 'timeout_error' in platform_data['common_failures']:
            base_config['element_timeout'] = 25  # Increase timeout
            base_config['retry_attempts'] = 5
        
        if 'element_not_found' in platform_data['common_failures']:
            base_config['use_multiple_strategies'] = True
            base_config['enable_ocr_fallback'] = True
        
        # Mobile-specific optimizations
        if platform == Platform.MOBILE:
            mobile_insights = self.mobile_specific_insights
            
            # CAPTCHA optimization
            if mobile_insights['captcha_success_rate']:
                avg_success = sum(mobile_insights['captcha_success_rate']) / len(mobile_insights['captcha_success_rate'])
                if avg_success < 0.8:
                    base_config['captcha_long_press_duration'] = 20000  # 20 seconds
                    base_config['captcha_retry_attempts'] = 5
                    base_config['captcha_coordinate_fallback'] = True
            
            # Gesture optimization
            best_gestures = {}
            for gesture_type, data in mobile_insights['gesture_optimization'].items():
                if data['total'] >= 3:
                    success_rate = data['success'] / data['total']
                    best_gestures[gesture_type] = success_rate
            
            if best_gestures:
                base_config['preferred_gestures'] = best_gestures
        
        return base_config

class TerminalExecutionEngine:
    """Enhanced terminal execution with intelligent monitoring and recovery"""
    
    def __init__(self, intelligence: ExecutionIntelligence):
        self.intelligence = intelligence
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.execution_logs: List[Dict[str, Any]] = []
        self.performance_monitor = PerformanceMonitor()
    
    def start_appium_server(self) -> bool:
        """Start Appium server with enhanced configuration"""
        try:
            # Check if Appium is already running
            if self._is_appium_running():
                self.intelligence.record_execution(Platform.MOBILE, {
                    'action': 'appium_server_start',
                    'status': 'already_running',
                    'timestamp': time.time()
                })
                return True
            
            # Kill any existing Appium processes
            self._kill_appium_processes()
            time.sleep(2)
            
            # Enhanced Appium server configuration
            appium_cmd = [
                "appium",
                "--port", "4723",
                "--session-override",
                "--local-timezone",
                "--log-timestamp",
                "--log-no-colors",
                "--relaxed-security",
                "--allow-insecure", "chromedriver_autodownload"
            ]
            
            # Start Appium server
            process = subprocess.Popen(
                appium_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True if os.name == 'nt' else False
            )
            
            self.active_processes['appium'] = process
            
            # Wait for server to be ready with intelligent timeout
            max_wait = 30
            for _ in range(max_wait):
                if self._is_appium_running():
                    self.intelligence.record_execution(Platform.MOBILE, {
                        'action': 'appium_server_start',
                        'status': 'success',
                        'startup_time': max_wait - _,
                        'timestamp': time.time()
                    })
                    return True
                time.sleep(1)
            
            # Server failed to start
            self.intelligence.record_execution(Platform.MOBILE, {
                'action': 'appium_server_start', 
                'status': 'failed',
                'error': 'timeout_waiting_for_server',
                'timestamp': time.time()
            })
            return False
            
        except Exception as e:
            self.intelligence.record_execution(Platform.MOBILE, {
                'action': 'appium_server_start',
                'status': 'failed', 
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def _is_appium_running(self) -> bool:
        """Check if Appium server is running with enhanced detection"""
        try:
            # Try multiple detection methods
            detection_methods = [
                # Method 1: Check port 4723
                lambda: self._check_port(4723),
                # Method 2: Check process list
                lambda: self._check_appium_process(),
                # Method 3: Try HTTP request to Appium
                lambda: self._check_appium_http()
            ]
            
            return any(method() for method in detection_methods)
            
        except Exception:
            return False
    
    def _check_port(self, port: int) -> bool:
        """Check if port is in use"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex(('localhost', port))
                return result == 0
        except:
            return False
    
    def _check_appium_process(self) -> bool:
        """Check if Appium process is running"""
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq node.exe'], 
                                      capture_output=True, text=True)
                return 'node.exe' in result.stdout and 'appium' in result.stdout.lower()
            else:  # Unix/Linux/Mac
                result = subprocess.run(['pgrep', '-f', 'appium'], capture_output=True, text=True)
                return bool(result.stdout.strip())
        except:
            return False
    
    def _check_appium_http(self) -> bool:
        """Check Appium server via HTTP request"""
        try:
            import urllib.request
            import urllib.error
            
            request = urllib.request.Request('http://localhost:4723/wd/hub/status')
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        except:
            return False
    
    def _kill_appium_processes(self):
        """Kill existing Appium processes"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', 'node.exe'], capture_output=True)
            else:  # Unix/Linux/Mac
                subprocess.run(['pkill', '-f', 'appium'], capture_output=True)
        except:
            pass
    
    def execute_script_with_monitoring(self, script_path: str, platform: Platform, 
                                     config: ExecutionConfig) -> Dict[str, Any]:
        """Execute script with comprehensive monitoring and recovery"""
        
        start_time = time.time()
        execution_id = f"{platform.value}_{int(start_time)}"
        
        # Get optimized configuration
        opt_config = self.intelligence.get_optimization_config(platform)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(execution_id)
        
        try:
            # Enhanced script execution with monitoring
            python_executable = self._get_python_executable()
            
            env = os.environ.copy()
            env.update({
                'PYTHONPATH': str(Path(script_path).parent),
                'AUTOMATION_EXECUTION_ID': execution_id,
                'OPTIMIZATION_CONFIG': json.dumps(opt_config)
            })
            
            # Execute with intelligent monitoring
            process = subprocess.Popen(
                [python_executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                shell=False
            )
            
            self.active_processes[execution_id] = process
            
            # Monitor execution with timeout handling
            try:
                stdout, stderr = process.communicate(timeout=config.timeout_seconds)
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                
                self.intelligence.record_execution(platform, {
                    'status': 'timeout',
                    'duration': time.time() - start_time,
                    'error': 'execution_timeout',
                    'timeout_seconds': config.timeout_seconds
                })
            
            # Stop performance monitoring
            perf_metrics = self.performance_monitor.stop_monitoring(execution_id)
            
            # Analyze execution results
            execution_data = {
                'duration': time.time() - start_time,
                'return_code': return_code,
                'stdout': stdout,
                'stderr': stderr,
                'performance_metrics': perf_metrics,
                'optimization_applied': list(opt_config.keys()),
                'device_info': self._get_device_info()
            }
            
            if return_code == 0:
                execution_data['status'] = 'completed'
                # Extract success metrics from stdout if available
                execution_data.update(self._parse_execution_output(stdout))
            else:
                execution_data['status'] = 'failed'
                execution_data['errors'] = self._parse_error_output(stderr)
            
            # Record for learning
            self.intelligence.record_execution(platform, execution_data)
            
            return {
                'status': 'success' if return_code == 0 else 'failed',
                'execution_id': execution_id,
                'return_code': return_code,
                'duration': execution_data['duration'],
                'stdout': stdout,
                'stderr': stderr,
                'performance_metrics': perf_metrics,
                'recommendations': self.intelligence.get_recommendations(platform),
                'optimization_applied': execution_data['optimization_applied']
            }
            
        except Exception as e:
            # Stop monitoring on exception
            perf_metrics = self.performance_monitor.stop_monitoring(execution_id)
            
            error_data = {
                'status': 'failed',
                'duration': time.time() - start_time,
                'error': str(e),
                'performance_metrics': perf_metrics,
                'device_info': self._get_device_info()
            }
            
            self.intelligence.record_execution(platform, error_data)
            
            return {
                'status': 'failed',
                'execution_id': execution_id,
                'error': str(e),
                'duration': error_data['duration'],
                'performance_metrics': perf_metrics,
                'recommendations': self.intelligence.get_recommendations(platform)
            }
        
        finally:
            # Cleanup
            if execution_id in self.active_processes:
                del self.active_processes[execution_id]
    
    def _get_python_executable(self) -> str:
        """Get appropriate Python executable"""
        # Try to use virtual environment if available
        venv_paths = [
            os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe'),  # Windows
            os.path.join(os.getcwd(), 'venv', 'bin', 'python'),  # Unix
            os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe'),  # Windows
            os.path.join(os.getcwd(), '.venv', 'bin', 'python')  # Unix
        ]
        
        for venv_path in venv_paths:
            if os.path.exists(venv_path):
                return venv_path
        
        return sys.executable
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information for mobile executions"""
        try:
            # Try ADB to get device info
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                devices = []
                for line in result.stdout.split('\n')[1:]:
                    if line.strip() and '\tdevice' in line:
                        devices.append(line.split('\t')[0])
                
                return {
                    'connected_devices': devices,
                    'primary_device': 'ZD222GXYPV' if 'ZD222GXYPV' in devices else devices[0] if devices else None,
                    'adb_available': True
                }
        except:
            pass
        
        return {'adb_available': False, 'connected_devices': [], 'primary_device': None}
    
    def _parse_execution_output(self, stdout: str) -> Dict[str, Any]:
        """Parse execution output for success metrics"""
        parsed_data = {}
        
        # Look for common success indicators
        if 'CAPTCHA' in stdout.upper():
            parsed_data['captcha_handled'] = 'SUCCESS' in stdout.upper()
        
        # Look for gesture information
        gesture_patterns = [
            r'(click|tap|swipe|long_press)\s+.*?(success|failed)',
            r'gesture\s+(\w+)\s+.*?(success|failed)'
        ]
        
        gestures_used = []
        for pattern in gesture_patterns:
            matches = re.finditer(pattern, stdout, re.IGNORECASE)
            for match in matches:
                gestures_used.append({
                    'type': match.group(1).lower(),
                    'success': 'success' in match.group(2).lower()
                })
        
        parsed_data['gestures_used'] = gestures_used
        
        # Look for step completion information
        step_pattern = r'step\s+(\d+).*?(completed|failed)'
        step_matches = re.finditer(step_pattern, stdout, re.IGNORECASE)
        completed_steps = sum(1 for match in step_matches if 'completed' in match.group(2).lower())
        parsed_data['step_count'] = completed_steps
        
        return parsed_data
    
    def _parse_error_output(self, stderr: str) -> List[Dict[str, Any]]:
        """Parse error output for failure analysis"""
        errors = []
        
        # Common error patterns
        error_patterns = [
            (r'TimeoutException', 'timeout_error'),
            (r'NoSuchElementException', 'element_not_found'),
            (r'ElementNotInteractableException', 'element_not_interactable'),
            (r'WebDriverException', 'webdriver_error'),
            (r'ConnectionRefusedError', 'connection_error'),
            (r'StaleElementReferenceException', 'stale_element')
        ]
        
        for pattern, error_type in error_patterns:
            if re.search(pattern, stderr, re.IGNORECASE):
                errors.append({
                    'type': error_type,
                    'pattern': pattern,
                    'context': stderr[:200]  # First 200 chars for context
                })
        
        return errors

class PerformanceMonitor:
    """Monitor execution performance metrics"""
    
    def __init__(self):
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
    
    def start_monitoring(self, execution_id: str):
        """Start monitoring execution"""
        self.active_monitors[execution_id] = {
            'start_time': time.time(),
            'cpu_start': self._get_cpu_usage(),
            'memory_start': self._get_memory_usage()
        }
    
    def stop_monitoring(self, execution_id: str) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        if execution_id not in self.active_monitors:
            return {}
        
        monitor_data = self.active_monitors[execution_id]
        end_time = time.time()
        
        metrics = {
            'execution_time': end_time - monitor_data['start_time'],
            'cpu_usage_start': monitor_data['cpu_start'],
            'cpu_usage_end': self._get_cpu_usage(),
            'memory_usage_start': monitor_data['memory_start'],
            'memory_usage_end': self._get_memory_usage()
        }
        
        # Calculate usage delta
        metrics['cpu_delta'] = metrics['cpu_usage_end'] - metrics['cpu_usage_start']
        metrics['memory_delta'] = metrics['memory_usage_end'] - metrics['memory_usage_start']
        
        del self.active_monitors[execution_id]
        return metrics
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0

# Validation Functions (Enhanced)
def validate_mobile_prerequisites_enhanced() -> Dict[str, bool]:
    """Enhanced mobile prerequisites validation"""
    results = {}
    
    # Check ADB
    try:
        result = subprocess.run(['adb', 'version'], capture_output=True, text=True, timeout=10)
        results['adb'] = result.returncode == 0
        if results['adb']:
            # Check for connected devices
            device_result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
            connected_devices = [line.split('\t')[0] for line in device_result.stdout.split('\n')[1:] 
                               if line.strip() and '\tdevice' in line]
            results['adb_devices'] = len(connected_devices) > 0
            results['target_device_connected'] = 'ZD222GXYPV' in connected_devices
    except:
        results['adb'] = False
        results['adb_devices'] = False
        results['target_device_connected'] = False
    
    # Check Appium
    try:
        result = subprocess.run(['appium', '--version'], capture_output=True, text=True, timeout=10)
        results['appium'] = result.returncode == 0
        
        # Check for modern gesture support
        if results['appium']:
            appium_version = result.stdout.strip()
            # Parse version to check for mobile: gesture support (available in 2.0+)
            version_match = re.search(r'(\d+)\.(\d+)', appium_version)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                results['modern_gestures'] = major >= 2
            else:
                results['modern_gestures'] = False
    except:
        results['appium'] = False
        results['modern_gestures'] = False
    
    # Check Python packages
    required_packages = ['appium-python-client', 'selenium', 'Pillow']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('Pillow', 'PIL'))
            results[f'python_{package}'] = True
        except ImportError:
            results[f'python_{package}'] = False
    
    # Check for OCR capabilities (for CAPTCHA handling)
    try:
        import pytesseract
        results['ocr_available'] = True
        # Check if tesseract executable is found
        try:
            pytesseract.get_tesseract_version()
            results['tesseract_executable'] = True
        except:
            results['tesseract_executable'] = False
    except ImportError:
        results['ocr_available'] = False
        results['tesseract_executable'] = False
    
    return results

def validate_web_prerequisites_enhanced() -> Dict[str, bool]:
    """Enhanced web prerequisites validation"""
    results = {}
    
    # Check Playwright
    try:
        import playwright
        results['playwright'] = True
        
        # Check for browser installations
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            results['chromium'] = p.chromium.executable_path is not None
            results['firefox'] = p.firefox.executable_path is not None
            results['webkit'] = p.webkit.executable_path is not None
    except ImportError:
        results['playwright'] = False
        results['chromium'] = False
        results['firefox'] = False
        results['webkit'] = False
    except Exception:
        results['playwright'] = True  # Playwright installed but browser check failed
        results['chromium'] = False
        results['firefox'] = False
        results['webkit'] = False
    
    return results

# Global instances
execution_intelligence = ExecutionIntelligence()

def run_enhanced_agent3(seq_no: str, script_path: str, config: ExecutionConfig) -> Dict[str, Any]:
    """
    Enhanced Agent 3: Script Execution with Intelligence
    
    MAINTAINS EXACT FUNCTION SIGNATURE - DO NOT CHANGE
    """
    try:
        start_time = time.time()
        
        # Validate inputs
        if not script_path or not Path(script_path).exists():
            raise HTTPException(status_code=400, detail=f"Script file not found: {script_path}")
        
        # Initialize execution engine
        engine = TerminalExecutionEngine(execution_intelligence)
        
        # Get intelligent recommendations before execution
        recommendations = execution_intelligence.get_recommendations(config.platform)
        
        # Platform-specific setup
        setup_success = True
        setup_logs = []
        
        if config.platform == Platform.MOBILE:
            # Enhanced mobile prerequisites validation
            mobile_prereqs = validate_mobile_prerequisites_enhanced()
            missing_prereqs = [k for k, v in mobile_prereqs.items() if not v]
            
            if missing_prereqs:
                setup_logs.append(f"Missing mobile prerequisites: {', '.join(missing_prereqs)}")
                
            # Start Appium server if needed
            if mobile_prereqs.get('appium', False):
                appium_started = engine.start_appium_server()
                if not appium_started:
                    setup_success = False
                    setup_logs.append("Failed to start Appium server")
            else:
                setup_success = False
                setup_logs.append("Appium not available")
                
        elif config.platform == Platform.WEB:
            # Enhanced web prerequisites validation  
            web_prereqs = validate_web_prerequisites_enhanced()
            missing_prereqs = [k for k, v in web_prereqs.items() if not v]
            
            if missing_prereqs:
                setup_logs.append(f"Missing web prerequisites: {', '.join(missing_prereqs)}")
                if not web_prereqs.get('playwright', False):
                    setup_success = False
        
        # Execute script with monitoring
        if setup_success:
            execution_result = engine.execute_script_with_monitoring(script_path, config.platform, config)
        else:
            execution_result = {
                'status': 'failed',
                'error': 'Prerequisites validation failed',
                'setup_logs': setup_logs
            }
        
        # Compile final results
        total_duration = time.time() - start_time
        
        result = {
            'status': 'success' if setup_success and execution_result.get('status') == 'success' else 'failed',
            'sequence_number': seq_no,
            'platform': config.platform.value,
            'total_duration': total_duration,
            'setup_duration': execution_result.get('duration', 0),
            'setup_success': setup_success,
            'setup_logs': setup_logs,
            'execution_result': execution_result,
            'recommendations': recommendations,
            'intelligence_stats': {
                'total_executions': len(execution_intelligence.execution_history),
                'platform_executions': len([e for e in execution_intelligence.execution_history if e['platform'] == config.platform.value]),
                'success_rate': len([e for e in execution_intelligence.execution_history if e['status'] == 'completed']) / max(1, len(execution_intelligence.execution_history))
            }
        }
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        execution_intelligence.record_execution(config.platform, {
            'status': 'failed',
            'error': error_msg,
            'duration': time.time() - start_time
        })
        
        raise HTTPException(status_code=500, detail=f"Agent 3 execution failed: {error_msg}")

def run_agent3(seq_no: str, script_path: str, platform: str = "mobile") -> Dict[str, Any]:
    """
    Simplified Agent 3 interface for backward compatibility
    
    MAINTAINS EXACT FUNCTION SIGNATURE - DO NOT CHANGE
    """
    try:
        # Convert platform string to enum
        platform_enum = Platform.MOBILE if platform.lower() == "mobile" else Platform.WEB
        
        # Create default config
        config = ExecutionConfig(
            platform=platform_enum,
            use_enhanced_agent=True,
            enable_monitoring=True,
            enable_reporting=True,
            timeout_seconds=300,
            retry_attempts=1
        )
        
        # Use enhanced agent 3
        return run_enhanced_agent3(seq_no, script_path, config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 3 execution failed: {str(e)}")

# Additional utility functions for monitoring and debugging
def get_execution_intelligence_stats() -> Dict[str, Any]:
    """Get intelligence system statistics"""
    return {
        'total_executions': len(execution_intelligence.execution_history),
        'mobile_executions': len([e for e in execution_intelligence.execution_history if e['platform'] == 'mobile']),
        'web_executions': len([e for e in execution_intelligence.execution_history if e['platform'] == 'web']),
        'success_rate': len([e for e in execution_intelligence.execution_history if e['status'] == 'completed']) / max(1, len(execution_intelligence.execution_history)),
        'mobile_insights': execution_intelligence.mobile_specific_insights,
        'platform_insights': {k.value: v for k, v in execution_intelligence.platform_insights.items()}
    }

def clear_execution_intelligence() -> None:
    """Clear intelligence data for troubleshooting"""
    execution_intelligence.execution_history.clear()
    execution_intelligence.performance_metrics.clear()
    execution_intelligence.common_issues.clear()
    execution_intelligence.success_patterns.clear()
    execution_intelligence.mobile_specific_insights = {
        'captcha_success_rate': [],
        'gesture_optimization': {},
        'device_performance': {},
        'element_detection_patterns': {}
    }

def get_platform_recommendations(platform: str, complexity_score: float = 0.5) -> List[str]:
    """Get recommendations for specific platform"""
    platform_enum = Platform.MOBILE if platform.lower() == "mobile" else Platform.WEB
    return execution_intelligence.get_recommendations(platform_enum, complexity_score)